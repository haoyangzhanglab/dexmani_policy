import torch
import torch.nn as nn
from einops import reduce
import torch.nn.functional as F
from dexmani_policy.agents.action_decoders.common.sample_util import SampleStrategy


class FlowMatch_With_Consistency(nn.Module):
    def __init__(
        self,
        model,
        denoise_timesteps = 10,
        flow_batch_ratio = 0.75,
        consistency_batch_ratio = 0.25,
        t_sample_mode_for_flow = "beta",
        t_sample_mode_for_consistency = "discrete",
        dt_sample_mode_for_consistency = "uniform",
        target_t_sample_mode = "relative",

    ):
        super().__init__()

        self.model = model
        self.denoise_timesteps = denoise_timesteps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "flow_batch_ratio and consistency_batch_ratio should sum to 1.0"

        # 更换默认采样策略有可能出现shape错误，尤其是dt
        self.t_sample_mode_for_flow = t_sample_mode_for_flow
        self.t_sample_mode_for_consistency = t_sample_mode_for_consistency
        self.dt_sample_mode_for_consistency = dt_sample_mode_for_consistency
        self.target_t_sample_mode = target_t_sample_mode    # relative: 输入DitX的是t, dt; absolute: 输入DitX的是t, t+dt
    
        self.sampler = SampleStrategy(denoise_timesteps=denoise_timesteps)
    

    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        interpolated_data_point = noise_coeff * noise + timestep * target
        return interpolated_data_point
    

    def get_flow_velocity(self, actions):
        '''
        生成流匹配的训练样本与监督信号, 即(x_t, t, dt) 和 v_target
        '''
        B = actions.shape[0]

        t_flow = self.sampler.sample(B, self.t_sample_mode_for_flow, device=actions.device)
        t_flow =  t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((B,), device=actions.device)

        if self.target_t_sample_mode == "relative":
            target_t_flow = dt_flow
        elif self.target_t_sample_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        
        x0_flow = torch.randn_like(actions, device=actions.device)
        x1_flow = actions               
        xt_flow = self.linear_interpolate(x0_flow, x1_flow, t_flow, epsilon=0.0)
        vt_flow_target = x1_flow - x0_flow

        flow_target_dict = {
            "xt": xt_flow,
            "t": t_flow,
            "dt": target_t_flow,
            "vt_target": vt_flow_target,
        }
        return flow_target_dict
    

    def get_consistency_velocity(self, actions, cond, ema_model):
        '''
        为一致性[consistency: ct]训练构造监督速度vt_target
        '''
        B = actions.shape[0]

        t_ct = self.sampler.sample(B, self.t_sample_mode_for_consistency, device=actions.device)
        t_ct = t_ct.view(-1, 1, 1)

        dt1 = self.sampler.sample(B, self.dt_sample_mode_for_consistency, device=actions.device)
        dt2 = dt1.clone() # 可以使用同样的dt, 也可以随机采样

        # 用更远一步的趋势，估计数据点大概在什么位置，把“现在到终点”的整体速度作为监督信号，指导模型学会更全局的趋势，而不是局部的切线趋势
        t_next = t_ct.squeeze() + dt1
        t_next = torch.clamp(t_next, max=1.0)
        t_next = t_next.view(-1, 1, 1)

        if self.target_t_sample_mode == "relative":
            target_t_next = dt2
        elif self.target_t_sample_mode == "absolute":
            target_t_next = t_next.squeeze() + dt2

        x0_ct = torch.randn_like(actions, device=actions.device)
        x1_ct = actions
        xt_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        xt_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # 使用EMA模型预测xt_next对应的vt, 作为一致性训练的监督信号
        # EMA模型在全局都应该设置为eval模式
        with torch.no_grad():
            v_avg_to_next_target = ema_model(
                x = xt_next,
                timestep = t_next.squeeze(),
                target_t = target_t_next,
                context = cond, 
            )
        
        pred_x1_ct = xt_next + v_avg_to_next_target * (1.0 - t_next)
        vt_target_ct = (pred_x1_ct - xt_ct) / (1.0 - t_ct)

        consistency_target_dict = {
            "xt": xt_ct,
            "t": t_ct,
            "dt": dt1 if self.target_t_sample_mode == "relative" else t_next.squeeze(),  # 输入DitX的dt
            "vt_target": vt_target_ct,
        }
        return consistency_target_dict



    def compute_loss(self, cond, actions, **kwargs):
        ema_model = kwargs.get("ema_model", None)
        assert ema_model is not None, "EMA model is required for computing consistency loss in FlowMatch_With_Consistency"

        B = actions.shape[0]
        flow_batchsize = int(B * self.flow_batch_ratio)

        loss = 0.

        #计算流匹配的监督信号和损失
        flow_targets = self.get_flow_velocity(actions[:flow_batchsize])
        pred_vt_flow = self.model(
            x = flow_targets["xt"],
            timestep = flow_targets["t"].squeeze(),
            target_t = flow_targets["dt"].squeeze(),
            context = cond[:flow_batchsize],
        )

        vt_flow_target = flow_targets["vt_target"]
        loss_flow = F.mse_loss(pred_vt_flow, vt_flow_target, reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean')
        loss += loss_flow.mean()
        loss_flow = loss_flow.mean()

        #计算一致性训练的监督信号和损失
        consistency_targets = self.get_consistency_velocity(actions[flow_batchsize:], cond[flow_batchsize:], ema_model)
        pred_vt_consistency = self.model(
            x = consistency_targets["xt"],
            timestep = consistency_targets["t"].squeeze(),
            target_t = consistency_targets["dt"].squeeze(),
            context = cond[flow_batchsize:],
        )

        vt_consistency_target = consistency_targets["vt_target"]
        loss_consistency = F.mse_loss(pred_vt_consistency, vt_consistency_target, reduction='none')
        loss_consistency = reduce(loss_consistency, 'b ... -> b (...)', 'mean')
        loss += loss_consistency.mean()
        loss_consistency = loss_consistency.mean()


        # 计算辅助指标
        pred_vt_flow_magnitude = torch.sqrt(torch.mean(pred_vt_flow ** 2))
        pred_vt_consistency_magnitude = torch.sqrt(torch.mean(pred_vt_consistency ** 2))

        # 返回loss
        loss = loss.mean()
        loss_dict = {
            "loss": loss,
            "loss_flow": loss_flow,
            "loss_consistency": loss_consistency,
            "pred_vt_flow_magnitude": pred_vt_flow_magnitude,
            "pred_vt_consistency_magnitude": pred_vt_consistency_magnitude,
        }

        return loss, loss_dict


    @torch.no_grad()
    def sample_ode(self, x0, N, cond):
        # traj: 流匹配采样过程中所有的采样结果X
        # detach：避免梯度传播，clone: 确保每一步的X都是独立的张量，避免原地修改导致的问题
        B = x0.shape[0]
        x = x0.clone()

        dt = 1.0 / N
        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N

        traj = []
        traj.append(x.clone())
        
        # Euler方法求解ODE，属于最简单的数值方法
        for i in range(N):
            ti = torch.ones((B,), device=x0.device) * t[i]
            if self.target_t_sample_mode == "relative":
                target_ti = torch.full((B,), dt, device=x0.device, dtype=x0.dtype)
            elif self.target_t_sample_mode == "absolute":
                target_ti = torch.clamp(ti + dt, max=1.0)
            vti_pred = self.model(
                x = x,
                timestep = ti,
                target_t = target_ti,
                context = cond,
            )
            x = x + vti_pred * dt
            traj.append(x.clone())
        
        return traj
    

    def predict_action(self, cond, action_template, denoise_timesteps=None):
        '''
        sample: zero_data for action seq, in order to get the shape and device
        '''       
        noise = torch.randn_like(action_template, device=action_template.device)

        if denoise_timesteps is None:
            denoise_timesteps = self.denoise_timesteps

        ode_traj = self.sample_ode(x0=noise, N=denoise_timesteps, cond=cond)
        return ode_traj[-1]


