import torch
import torch.nn as nn
from einops import reduce
import torch.nn.functional as F
from dexmani_policy.agents.action_decoders.common.sample import SampleLibrary


class FlowMatchWithConsistency(nn.Module):
    def __init__(
        self,
        model,
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        consistency_batch_ratio: float = 0.25,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",

    ):
        super().__init__()

        self.model = model
        self.denoise_timesteps = denoise_timesteps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert abs(flow_batch_ratio + consistency_batch_ratio - 1.0) < 1e-6, "flow_batch_ratio and consistency_batch_ratio should sum to 1.0"

        # Changing the default sampling strategy may cause shape errors, especially for dt.
        self.t_sample_mode_for_flow = t_sample_mode_for_flow
        self.t_sample_mode_for_consistency = t_sample_mode_for_consistency
        self.dt_sample_mode_for_consistency = dt_sample_mode_for_consistency
        self.target_t_sample_mode = target_t_sample_mode    # relative: input DiT-X with (t, dt); absolute: input (t, t+dt)
    
        self.sampler = SampleLibrary(denoise_timesteps=denoise_timesteps)
    

    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        interpolated_data_point = noise_coeff * noise + timestep * target
        return interpolated_data_point
    

    def get_flow_velocity(self, actions):
        B = actions.shape[0]

        t_flow = self.sampler.sample(B, self.t_sample_mode_for_flow, device=actions.device)
        t_flow =  t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((B,), device=actions.device)

        if self.target_t_sample_mode == "relative":
            target_t_flow = dt_flow
        elif self.target_t_sample_mode == "absolute":
            target_t_flow = torch.clamp(t_flow.squeeze() + dt_flow, max=1.0)
        
        x0_flow = torch.randn_like(actions, device=actions.device)
        x1_flow = actions               
        xt_flow = self.linear_interpolate(x0_flow, x1_flow, t_flow, epsilon=0.0)
        vt_flow_target = x1_flow - x0_flow

        flow_target_dict = {
            "xt": xt_flow,
            "t": t_flow,
            "target_t": target_t_flow,
            "vt_target": vt_flow_target,
        }
        return flow_target_dict
    

    def get_consistency_velocity(self, actions, cond, ema_model):
        B = actions.shape[0]

        t_ct = self.sampler.sample(B, self.t_sample_mode_for_consistency, device=actions.device)
        t_ct = t_ct.view(-1, 1, 1)

        dt1 = self.sampler.sample(B, self.dt_sample_mode_for_consistency, device=actions.device)
        # 简化实现：学生和教师使用相同的预测步长 dt（可根据需要独立采样 dt2 以解耦）
        dt2 = dt1.clone()

        # 用更远一步的趋势，估计数据点大概在什么位置，把“现在到终点”的整体速度作为监督信号，指导模型学会更全局的趋势，而不是局部的切线趋势
        t_next = t_ct.squeeze() + dt1
        t_next = torch.clamp(t_next, max=1.0)
        t_next = t_next.view(-1, 1, 1)

        if self.target_t_sample_mode == "relative":
            target_t_next = dt2
        elif self.target_t_sample_mode == "absolute":
            target_t_next = torch.clamp(t_next.squeeze() + dt2, max=1.0)

        x0_ct = torch.randn_like(actions, device=actions.device)
        x1_ct = actions
        xt_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        xt_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # Predict v_t at xt_next using the EMA model as consistency target.
        # The EMA model should always be in eval mode globally.
        with torch.no_grad():
            v_avg_to_next_target = ema_model(
                x = xt_next,
                timestep = t_next.squeeze(),
                target_t = target_t_next,
                context = cond, 
            )
        
        pred_x1_ct = xt_next + v_avg_to_next_target * (1.0 - t_next)
        vt_target_ct = (pred_x1_ct - xt_ct) / (1.0 - t_ct).clamp(min=1e-5)

        consistency_target_dict = {
            "xt": xt_ct,
            "t": t_ct,
            # relative 模式下，教师调用时传入的是 target_t_next=dt2，此处学生侧传入 dt1。
            # 由于 dt2 = dt1.clone()，数值相同，无实际影响。
            # 若将来解耦 dt1/dt2（独立采样），此处应改为 dt2 以保持语义一致。
            "target_t": dt1 if self.target_t_sample_mode == "relative" else t_next.squeeze(),
            "vt_target": vt_target_ct,
        }
        return consistency_target_dict



    def compute_loss(self, cond, actions, **kwargs):
        ema_model = kwargs.get("ema_model", None)

        B = actions.shape[0]
        if B < 2:
            zero = actions.sum() * 0.0
            return zero, {
                "loss": zero,
                "loss_flow": zero,
                "loss_consistency": zero,
                "pred_vt_flow_magnitude": zero,
            }

        flow_batchsize = max(1, min(B - 1, int(B * self.flow_batch_ratio)))

        flow_targets = self.get_flow_velocity(actions[:flow_batchsize])
        pred_vt_flow = self.model(
            x = flow_targets["xt"],
            timestep = flow_targets["t"].squeeze(),
            target_t = flow_targets["target_t"].squeeze(),
            context = cond[:flow_batchsize],
        )
        vt_flow_target = flow_targets["vt_target"]
        loss_flow = F.mse_loss(pred_vt_flow, vt_flow_target, reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean').mean()

        if ema_model is None:
            loss_dict = {
                "loss": loss_flow,
                "loss_flow": loss_flow,
                "loss_consistency": torch.zeros_like(loss_flow),
                "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_vt_flow ** 2)),
            }
            return loss_flow, loss_dict

        consistency_targets = self.get_consistency_velocity(actions[flow_batchsize:], cond[flow_batchsize:], ema_model)
        pred_vt_consistency = self.model(
            x = consistency_targets["xt"],
            timestep = consistency_targets["t"].squeeze(),
            target_t = consistency_targets["target_t"].squeeze(),
            context = cond[flow_batchsize:],
        )
        vt_consistency_target = consistency_targets["vt_target"]
        loss_consistency = F.mse_loss(pred_vt_consistency, vt_consistency_target, reduction='none')
        loss_consistency = reduce(loss_consistency, 'b ... -> b (...)', 'mean').mean()

        loss = loss_flow + loss_consistency
        loss_dict = {
            "loss": loss,
            "loss_flow": loss_flow,
            "loss_consistency": loss_consistency,
            "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_vt_flow ** 2)),
            "pred_vt_consistency_magnitude": torch.sqrt(torch.mean(pred_vt_consistency ** 2)),
        }
        return loss, loss_dict


    @torch.no_grad()
    def sample_ode(self, x0, N, cond):
        B = x0.shape[0]
        x = x0.clone()

        dt = 1.0 / N
        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N

        traj = []
        traj.append(x.clone())

        # Euler integration: x_{t+dt} = x_t + v_t * dt
        for i in range(N):
            ti = torch.ones((B,), device=x0.device) * t[i]
            if self.target_t_sample_mode == "relative":
                target_ti = torch.full((B,), dt, device=x0.device, dtype=x0.dtype)
            elif self.target_t_sample_mode == "absolute":
                target_ti = torch.clamp(ti + dt, max=1.0)
            # vti_pred: (B, horizon, action_dim)
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
        noise = torch.randn_like(action_template, device=action_template.device)

        if denoise_timesteps is None:
            denoise_timesteps = self.denoise_timesteps

        ode_traj = self.sample_ode(x0=noise, N=denoise_timesteps, cond=cond)
        return ode_traj[-1]


