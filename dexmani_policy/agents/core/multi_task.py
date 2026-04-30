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
    多任务 Agent 基类

    在 BaseAgent 基础上添加 task conditioning 支持。
    Task embedding 可以通过以下方式注入：
        1. 拼接到 obs_encoder 输出的 cond 向量
        2. 作为额外的 FiLM 条件传入 action_decoder
        3. 通过 cross-attention 注入（如果 backbone 支持）
    """

    def __init__(
        self,
        base_agent: nn.Module,
        num_tasks: int,
        task_embedding_dim: int = 64,
        task_cond_mode: str = 'concat',
        task_dropout: float = 0.1,
    ):
        super().__init__()
        # 只支持 concat 模式，film 和 cross_attn 需要修改 action_decoder
        if task_cond_mode != 'concat':
            raise ValueError(
                f"Only 'concat' mode is currently supported, got '{task_cond_mode}'. "
                f"'film' and 'cross_attn' modes require action_decoder modifications."
            )

        self.base_agent = base_agent
        self.num_tasks = num_tasks
        self.task_cond_mode = task_cond_mode

        # Task embedding
        self.task_embedding = TaskEmbedding(
            num_tasks=num_tasks,
            embedding_dim=task_embedding_dim,
            dropout=task_dropout,
        )

        # 继承 base_agent 的属性
        self.obs_encoder = base_agent.obs_encoder
        self.action_decoder = base_agent.action_decoder
        self.normalizer = base_agent.normalizer
        self.horizon = base_agent.horizon
        self.n_obs_steps = base_agent.n_obs_steps
        self.n_action_steps = base_agent.n_action_steps
        self.action_dim = base_agent.action_dim

    def load_normalizer_from_dataset(self, normalizer):
        self.base_agent.load_normalizer_from_dataset(normalizer)
        self.normalizer = self.base_agent.normalizer

    def preprocess(self, obs_dict):
        return self.base_agent.preprocess(obs_dict)

    def _inject_task_condition(self, cond: torch.Tensor, task_emb: torch.Tensor):
        """将 task embedding 拼接到 condition 中

        Args:
            cond: (B, cond_dim) 观测条件向量
            task_emb: (B, task_emb_dim) 任务嵌入向量

        Returns:
            (B, cond_dim + task_emb_dim) 拼接后的条件向量
        """
        return torch.cat([cond, task_emb], dim=-1)

    def compute_loss(self, batch, **kwargs):
        # 提取 task_id
        task_ids = batch.get('task_id')
        if task_ids is None:
            raise ValueError("batch must contain 'task_id' for MultiTaskAgent")

        # 编码观测
        cond = self.obs_encoder(self.preprocess(batch['obs']))

        # 编码任务
        task_emb = self.task_embedding(task_ids)

        # 注入任务条件
        cond_with_task = self._inject_task_condition(cond, task_emb)

        # 计算损失
        nactions = self.normalizer['action'].normalize(batch['action'])
        return self.action_decoder.compute_loss(cond_with_task, nactions, **kwargs)

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
        cond = self.obs_encoder(self.preprocess(obs_dict))

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
        pred = self.action_decoder.predict_action(cond_with_task, template, denoise_timesteps)
        pred = self.normalizer['action'].unnormalize(pred)

        s = self.n_obs_steps - 1
        return {
            'pred_action': pred,
            'control_action': pred[:, s:s + self.n_action_steps],
        }

    def configure_optimizer(self, **kwargs):
        # 将 task_embedding 参数加入优化器
        base_optimizer = self.base_agent.configure_optimizer(**kwargs)

        # 添加 task_embedding 参数组
        task_params = list(self.task_embedding.parameters())
        if task_params:
            base_optimizer.add_param_group({
                'params': task_params,
                'lr': kwargs.get('lr', 1e-4),
                'weight_decay': kwargs.get('weight_decay', 1e-6),
            })

        return base_optimizer
