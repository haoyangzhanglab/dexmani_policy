import torch
import torch.nn as nn
from einops import reduce
import torch.nn.functional as F
from dexmani_policy.agents.action_decoders.sample import TimeSampler


class FlowMatchWithConsistency(nn.Module):
    """Flow Matching decoder with optional consistency training.

    Combines two complementary training objectives:

    **Flow Matching**: predicts the velocity field v = x1 - x0 along rectified
    flow straight-line paths x_t = (1-t)*x0 + t*x1. Uses the first
    ``flow_batch_ratio`` fraction of each batch.

    **Consistency Training** (when EMA teacher is available): the EMA teacher
    estimates pred_x1 at t_next, the target velocity is derived as
    (pred_x1 - x_t) / (1-t), and the student matches it via MSE. At t_next=1.0
    (≈45% of samples) the teacher target degenerates to the exact x1-x0.

    The two sub-batches are merged into one forward pass to keep
    ``torch.compile`` happy with a fixed batch size.

    Parameters:
        model: Backbone (DiTXFlowMatch) that predicts velocity given (x, t, target_t, context).
        denoise_timesteps: Number of Euler integration steps at inference.
        flow_batch_ratio: Fraction of batch used for flow loss (remainder for consistency).
        target_t_sample_mode: ``"relative"`` — model receives dt as target_t;
            ``"absolute"`` — model receives t+dt as target_t.
    """
    def __init__(
        self,
        model: nn.Module,
        denoise_timesteps: int = 10,
        flow_batch_ratio: float = 0.75,
        t_sample_mode_for_flow: str = "beta",
        t_sample_mode_for_consistency: str = "discrete",
        dt_sample_mode_for_consistency: str = "uniform",
        target_t_sample_mode: str = "relative",
    ) -> None:
        super().__init__()

        self.model = model
        self.denoise_timesteps = denoise_timesteps
        self.flow_batch_ratio = flow_batch_ratio

        self.t_sample_mode_for_flow = t_sample_mode_for_flow
        self.t_sample_mode_for_consistency = t_sample_mode_for_consistency
        self.dt_sample_mode_for_consistency = dt_sample_mode_for_consistency
        self.target_t_sample_mode = target_t_sample_mode    # relative: input DiT-X with (t, dt); absolute: input (t, t+dt)
    
        self.sampler = TimeSampler(denoise_timesteps=denoise_timesteps)
    

    def linear_interpolate(
        self, noise: torch.Tensor, target: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        noise_coeff = 1.0 - timestep
        interpolated_data_point = noise_coeff * noise + timestep * target
        return interpolated_data_point
    

    def get_flow_velocity(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
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
        xt_flow = self.linear_interpolate(x0_flow, x1_flow, t_flow)
        vt_flow_target = x1_flow - x0_flow

        flow_target_dict = {
            "xt": xt_flow,
            "t": t_flow,
            "target_t": target_t_flow,
            "vt_target": vt_flow_target,
        }
        return flow_target_dict
    

    def get_consistency_velocity(
        self, actions: torch.Tensor, cond: torch.Tensor, ema_model: nn.Module
    ) -> dict[str, torch.Tensor]:
        # target_t 作为 mode indicator：
        #   target_t=0 (flow): 预测瞬时速度 v = x1-x0
        #   target_t>0 (consistency): 预测到 x1 的有效速度
        # 对于 rectified flow 的直线路径，两者等价 (= x1-x0)。
        # consistency training 在速度场不完美时提供自校正信号。
        B = actions.shape[0]

        t_ct = self.sampler.sample(B, self.t_sample_mode_for_consistency, device=actions.device)
        t_ct = t_ct.view(-1, 1, 1)

        dt1 = self.sampler.sample(B, self.dt_sample_mode_for_consistency, device=actions.device)
        dt2 = dt1.clone()

        t_next = t_ct.squeeze() + dt1
        t_next = torch.clamp(t_next, max=1.0)
        t_next = t_next.view(-1, 1, 1)

        if self.target_t_sample_mode == "relative":
            target_t_next = dt2
        elif self.target_t_sample_mode == "absolute":
            target_t_next = torch.clamp(t_next.squeeze() + dt2, max=1.0)

        x0_ct = torch.randn_like(actions, device=actions.device)
        x1_ct = actions
        xt_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct)
        xt_next = self.linear_interpolate(x0_ct, x1_ct, t_next)

        with torch.no_grad():
            v_avg_to_next_target = ema_model(
                x = xt_next,
                timestep = t_next.squeeze(),
                target_t = target_t_next,
                context = cond, 
            )
        
        pred_x1_ct = xt_next + v_avg_to_next_target * (1.0 - t_next)
        # Clamp aligned with discrete sampling worst case (1/denoise_timesteps);
        # floor of 1e-3 protects continuous sampling modes (uniform/lognorm).
        vt_target_ct = (pred_x1_ct - xt_ct) / (1.0 - t_ct).clamp(min=max(1.0 / self.denoise_timesteps, 1e-3))

        consistency_target_dict = {
            "xt": xt_ct,
            "t": t_ct,
            "target_t": dt2 if self.target_t_sample_mode == "relative" else target_t_next.squeeze(),
            "vt_target": vt_target_ct,
        }
        return consistency_target_dict



    def compute_loss(
        self,
        cond: torch.Tensor,
        actions: torch.Tensor,
        ema_backbone: nn.Module | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ema_model = ema_backbone

        B = actions.shape[0]

        # 无 EMA teacher 时全量样本用于 flow loss，避免无效的数据拆分浪费
        if ema_model is None:
            flow_targets = self.get_flow_velocity(actions)
            pred_vt_flow = self.model(
                x = flow_targets["xt"],
                timestep = flow_targets["t"].squeeze(),
                target_t = flow_targets["target_t"].squeeze(),
                context = cond,
            )
            loss_flow = F.mse_loss(pred_vt_flow, flow_targets["vt_target"], reduction='none')
            loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean').mean()
            loss_dict = {
                "loss": loss_flow,
                "loss_action": loss_flow,
                "loss_flow": loss_flow,
                "loss_consistency": torch.zeros_like(loss_flow),
                "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_vt_flow ** 2)),
                "has_consistency": 0,
            }
            return loss_flow, loss_dict

        # 检查 batch size 是否足够进行 consistency 训练
        if B < 2:
            import warnings
            warnings.warn(
                f"FlowMatch: batch_size={B} < 2, consistency training disabled for this batch "
                f"(only flow loss will be computed). "
                f"If this warning appears frequently during training, consider increasing "
                f"dataloader.batch_size to ensure batch_size >= 2 for better target_t generalization.",
                UserWarning,
                stacklevel=2
            )

        flow_batchsize = max(1, min(B - 1, int(B * self.flow_batch_ratio)))
        consistency_batchsize = B - flow_batchsize

        flow_targets = self.get_flow_velocity(actions[:flow_batchsize])

        # consistency_batchsize == 0: 降级为纯 flow loss
        if consistency_batchsize == 0:
            pred_vt_flow = self.model(
                x=flow_targets["xt"],
                timestep=flow_targets["t"].squeeze(),
                target_t=flow_targets["target_t"].squeeze(),
                context=cond[:flow_batchsize],
            )
            loss_flow = F.mse_loss(pred_vt_flow, flow_targets["vt_target"], reduction='none')
            loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean').mean()
            loss_dict = {
                "loss": loss_flow, "loss_action": loss_flow,
                "loss_flow": loss_flow, "loss_consistency": torch.zeros_like(loss_flow),
                "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_vt_flow ** 2)),
                "has_consistency": 0,
            }
            return loss_flow, loss_dict

        consistency_targets = self.get_consistency_velocity(actions[flow_batchsize:], cond[flow_batchsize:], ema_model)

        # 沿 batch 维拼接 flow 和 consistency 的 student 输入，合并为一次 forward，
        # 保证 torch.compile 始终看到固定 batch size，避免 stride guard 崩溃。
        x_merged = torch.cat([flow_targets["xt"], consistency_targets["xt"]], dim=0)
        t_merged = torch.cat([flow_targets["t"].reshape(-1), consistency_targets["t"].reshape(-1)], dim=0)
        target_t_merged = torch.cat([flow_targets["target_t"].reshape(-1),
                                      consistency_targets["target_t"].reshape(-1)], dim=0)

        pred_merged = self.model(
            x=x_merged,
            timestep=t_merged,
            target_t=target_t_merged,
            context=cond,
        )

        pred_vt_flow = pred_merged[:flow_batchsize]
        pred_vt_consistency = pred_merged[flow_batchsize:]

        loss_flow = F.mse_loss(pred_vt_flow, flow_targets["vt_target"], reduction='none')
        loss_flow = reduce(loss_flow, 'b ... -> b (...)', 'mean').mean()

        loss_consistency = F.mse_loss(pred_vt_consistency, consistency_targets["vt_target"], reduction='none')
        loss_consistency = reduce(loss_consistency, 'b ... -> b (...)', 'mean').mean()

        loss = loss_flow + loss_consistency
        loss_dict = {
            "loss": loss,
            "loss_action": loss,
            "loss_flow": loss_flow,
            "loss_consistency": loss_consistency,
            "pred_vt_flow_magnitude": torch.sqrt(torch.mean(pred_vt_flow ** 2)),
            "pred_vt_consistency_magnitude": torch.sqrt(torch.mean(pred_vt_consistency ** 2)),
            "flow_batch_size": flow_batchsize,
            "consistency_batch_size": consistency_batchsize,
            "t_flow_mean": flow_targets["t"].mean(),
            "t_consistency_mean": consistency_targets["t"].mean(),
            "has_consistency": 1,
        }
        return loss, loss_dict


    @torch.no_grad()
    def sample_ode(
        self,
        x0: torch.Tensor,
        N: int,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Euler 积分采样。

        relative 模式下 target_t=dt（>0），依赖 consistency 训练提供 target_t 泛化。
        若不启用 consistency（use_ema_teacher_for_consistency=False），模型仅在
        target_t=0 下训练，推理时的 target_t>0 会导致条件分布偏移。
        """
        B = x0.shape[0]
        x = x0

        dt = 1.0 / N
        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N

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

        return x


    def predict_action(
        self,
        cond: torch.Tensor,
        action_template: torch.Tensor,
        denoise_timesteps: int | None = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_template, device=action_template.device)

        if denoise_timesteps is None:
            denoise_timesteps = self.denoise_timesteps

        ode_traj = self.sample_ode(x0=noise, N=denoise_timesteps, cond=cond)
        return ode_traj


class FlowMatch(nn.Module):
    """纯 Flow Matching decoder（无 consistency training）。

    使用与 Diffusion 相同的 backbone（DiT_Diffusion），仅预测目标从 noise/sample
    变为 velocity field v = x1 - x0。

    训练: 沿直线路径 x_t = t*x1 + (1-t)*x0 预测速度场，MSE loss。
    推理: Euler ODE 积分 x_{t+dt} = x_t + v_θ(x_t, t) * dt。
    """

    def __init__(
        self,
        model,
        num_inference_steps: int = 10,
        t_sample_mode: str = "beta",
        beta_s: float = 0.999,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
    ):
        super().__init__()
        self.model = model
        self.num_inference_steps = num_inference_steps

        self.sampler = TimeSampler(
            denoise_timesteps=num_inference_steps,
            beta_s=beta_s,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
        )
        self.t_sample_mode = t_sample_mode

    def compute_loss(self, cond, actions, **kwargs):
        B = actions.shape[0]

        x0 = torch.randn_like(actions, device=actions.device)
        x1 = actions
        t = self.sampler.sample(B, self.t_sample_mode, device=actions.device)
        t = t.view(-1, 1, 1)

        xt = (1.0 - t) * x0 + t * x1

        target_v = x1 - x0

        pred_v = self.model(
            x=xt,
            timestep=t.squeeze(),
            context=cond,
        )

        loss = F.mse_loss(pred_v, target_v)
        loss_dict = {
            "loss": loss,
            "loss_action": loss,
            "pred_v_magnitude": torch.sqrt(torch.mean(pred_v ** 2)),
        }
        return loss, loss_dict

    @torch.no_grad()
    def predict_action(self, cond, action_template, denoise_timesteps=None):
        B = action_template.shape[0]
        device = action_template.device

        if denoise_timesteps is None:
            denoise_timesteps = self.num_inference_steps

        x = torch.randn_like(action_template, device=device)
        dt = 1.0 / denoise_timesteps

        for i in range(denoise_timesteps):
            ti = torch.full((B,), i * dt, device=device, dtype=x.dtype)
            v = self.model(x=x, timestep=ti, context=cond)
            x = x + v * dt

        return x

