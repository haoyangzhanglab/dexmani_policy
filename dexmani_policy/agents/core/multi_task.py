import torch
import torch.nn as nn
from typing import Optional


class TaskEmbedding(nn.Module):
    """
    任务嵌入模块，将离散的 task_id 映射为连续向量

    用于多任务学习中的任务条件化。
    """

    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_tasks, embedding_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: (B,) 或 (B, 1) 的整数 tensor

        Returns:
            task_emb: (B, embedding_dim)
        """
        if task_ids.dim() == 2:
            task_ids = task_ids.squeeze(-1)

        task_emb = self.embedding(task_ids)
        task_emb = self.dropout(task_emb)
        return task_emb


class MultiTaskAgent(nn.Module):
    """
    多任务 Agent，使用 FiLM 调制注入任务条件

    FiLM (Feature-wise Linear Modulation) 通过 scale 和 shift 参数调制观测特征：
        modulated_cond = cond * (1 + scale) + shift

    优点：
        1. 不修改网络输入维度，兼容所有 backbone（UNet1D, DiTX）
        2. 符合多任务模仿学习社区主流实践（MT-ACT, RoboAgent）
        3. 比 concat 模式更灵活，任务信息可以非线性调制特征

    Args:
        base_agent: 基础单任务 agent（DP3Agent, ManiFlowAgent 等）
        num_tasks: 任务数量
        task_embedding_dim: 任务嵌入维度
        task_dropout: 任务嵌入的 dropout 概率
        obs_cond_dim: 观测条件维度（可选）。如果为 None，将通过前向传播推断。
                      显式指定可以避免初始化时的推断开销和潜在错误。
    """

    def __init__(
        self,
        base_agent: nn.Module,
        num_tasks: int,
        task_embedding_dim: int = 64,
        task_dropout: float = 0.1,
        obs_cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.base_agent = base_agent
        self.num_tasks = num_tasks

        # Task embedding
        self.task_embedding = TaskEmbedding(
            num_tasks=num_tasks,
            embedding_dim=task_embedding_dim,
            dropout=task_dropout,
        )

        # 继承 base_agent 的标量属性（非 nn.Module，不会触发重复注册）
        self.horizon = base_agent.horizon
        self.n_obs_steps = base_agent.n_obs_steps
        self.n_action_steps = base_agent.n_action_steps
        self.action_dim = base_agent.action_dim

        # 获取 obs_encoder 输出维度
        if obs_cond_dim is None:
            raise ValueError(
                "obs_cond_dim must be explicitly specified in the config. "
                "For film mode: obs_cond_dim = n_obs_steps * (pc_out_dim + state_out_dim). "
                "For cross_attention mode: obs_cond_dim = token_dim (per-token dimension)."
            )

        self.obs_cond_dim = obs_cond_dim

        # 构建 FiLM generator
        self.film_generator = nn.Sequential(
            nn.Linear(task_embedding_dim, task_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(task_embedding_dim * 2, obs_cond_dim * 2),  # scale + shift
        )

        # 初始化 FiLM generator：scale 初始化为 0，shift 初始化为 0
        # 这样初始时 FiLM 是恒等变换，不影响预训练的 base_agent
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)

    @property
    def normalizer(self):
        return self.base_agent.normalizer

    def load_normalizer_from_dataset(self, normalizer):
        self.base_agent.load_normalizer_from_dataset(normalizer)

    def preprocess(self, obs_dict):
        return self.base_agent.preprocess(obs_dict)

    def _inject_task_condition(self, cond: torch.Tensor, task_emb: torch.Tensor):
        """使用 FiLM 调制注入任务条件

        Args:
            cond: 观测条件
                - UNet: (B, cond_dim)
                - DiT: (B, num_tokens, token_dim)
            task_emb: (B, task_emb_dim) 任务嵌入向量

        Returns:
            modulated_cond: FiLM 调制后的条件，形状与 cond 相同
        """
        # 生成 FiLM 参数
        film_params = self.film_generator(task_emb)  # (B, cond_dim * 2)
        scale, shift = film_params.chunk(2, dim=-1)  # (B, cond_dim) each

        # 根据 cond 的形状应用 FiLM
        if cond.dim() == 2:  # UNet: (B, D)
            assert cond.shape[-1] == self.obs_cond_dim, (
                f"FiLM dim mismatch: cond has {cond.shape[-1]} but obs_cond_dim={self.obs_cond_dim}"
            )
            return cond * (1 + scale) + shift
        elif cond.dim() == 3:  # DiT: (B, T, D)
            assert cond.shape[-1] == self.obs_cond_dim, (
                f"FiLM dim mismatch for cross_attention: cond last dim={cond.shape[-1]} "
                f"but obs_cond_dim={self.obs_cond_dim}. "
                f"For cross_attention mode, obs_cond_dim should be per-token dim, not flattened dim."
            )
            return cond * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected cond shape: {cond.shape}")

    def compute_loss(self, batch, **kwargs):
        # 提取 task_id
        task_ids = batch.get('task_id')
        if task_ids is None:
            raise ValueError("batch must contain 'task_id' for MultiTaskAgent")

        # 编码观测
        cond = self.base_agent.obs_encoder(self.preprocess(batch['obs']))

        # 编码任务
        task_emb = self.task_embedding(task_ids)

        # 注入任务条件
        cond_with_task = self._inject_task_condition(cond, task_emb)

        # 计算损失
        nactions = self.base_agent.normalizer['action'].normalize(batch['action'])

        # 提取 EMA backbone（兼容 FlowMatch consistency distillation）
        loss_kwargs = dict(kwargs)
        ema_model = loss_kwargs.pop('ema_model', None)
        if ema_model is not None:
            if hasattr(ema_model, 'base_agent'):
                ema_backbone = ema_model.base_agent.action_decoder.model
            elif hasattr(ema_model, 'action_decoder'):
                ema_backbone = ema_model.action_decoder.model
            else:
                ema_backbone = ema_model
            loss_kwargs['ema_model'] = ema_backbone

        return self.base_agent.action_decoder.compute_loss(cond_with_task, nactions, **loss_kwargs)

    @torch.no_grad()
    def predict_action(self, obs_dict, task_id: Optional[int] = None, denoise_timesteps=None):
        """
        Args:
            obs_dict: 观测字典
            task_id: 任务 ID，范围 [0, num_tasks)，默认为 0
            denoise_timesteps: 去噪步数
        """
        if task_id is None:
            task_id = 0

        # 检查 task_id 范围
        if not (0 <= task_id < self.num_tasks):
            raise ValueError(
                f"task_id must be in range [0, {self.num_tasks}), got {task_id}. "
                f"Available task IDs: {list(range(self.num_tasks))}"
            )

        # 编码观测
        cond = self.base_agent.obs_encoder(self.preprocess(obs_dict))

        # 编码任务
        batch_size = cond.shape[0]
        task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=cond.device)
        task_emb = self.task_embedding(task_ids)

        # 注入任务条件
        cond_with_task = self._inject_task_condition(cond, task_emb)

        # 预测动作
        template = torch.zeros(
            batch_size, self.horizon, self.action_dim,
            device=cond.device, dtype=cond.dtype,
        )
        pred = self.base_agent.action_decoder.predict_action(cond_with_task, template, denoise_timesteps)
        pred = self.base_agent.normalizer['action'].unnormalize(pred)

        s = self.n_obs_steps - 1
        return {
            'pred_action': pred,
            'control_action': pred[:, s:s + self.n_action_steps],
        }

    def configure_optimizer(self, **kwargs):
        base_optimizer = self.base_agent.configure_optimizer(**kwargs)

        # 添加 task_embedding + film_generator 参数组
        multi_task_params = (
            list(self.task_embedding.parameters())
            + list(self.film_generator.parameters())
        )
        if multi_task_params:
            base_optimizer.add_param_group({
                'params': multi_task_params,
                'lr': kwargs.get('lr', 1e-4),
                'weight_decay': kwargs.get('weight_decay', 1e-6),
            })

        return base_optimizer
