import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)


import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.common.pytorch_util import set_seed
from dexmani_policy.common.pytorch_util import worker_init_fn
from dexmani_policy.common.resolver import register_resolvers
from dexmani_policy.training.trainer import Trainer
from dexmani_policy.training.common.workspace import TrainWorkspace
from dexmani_policy.training.common.lr_scheduler import get_scheduler

register_resolvers()


def build_dataset_and_normalizer(cfg, rank=None, world_size=None):
    if rank is not None:
        # DDP 模式：DistributedSampler 负责数据分片，dataset 层面使用相同 seed
        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        dataset = hydra.utils.instantiate(dataset_cfg)
    else:
        # 单卡模式：直接实例化
        dataset = hydra.utils.instantiate(cfg.dataset)

    normalizer = dataset.get_normalizer()
    if hasattr(dataset, 'normalizer_mode') and dataset.normalizer_mode == 'per_task':
        raise NotImplementedError(
            "normalizer_mode='per_task' requires per-task normalizer loading, "
            "which is not yet integrated into the standard training entry. "
            "Use normalizer_mode='shared' or call get_normalizer(task_name=...) manually."
        )
    return dataset, normalizer


def build_model_and_ema(cfg, device, normalizer):
    model = hydra.utils.instantiate(cfg.agent)
    model.load_normalizer_from_dataset(normalizer)
    model.to(device)

    ema_model = None
    ema_updater = None
    if cfg.training.use_ema:
        ema_model = hydra.utils.instantiate(cfg.agent)
        ema_model.load_normalizer_from_dataset(normalizer)
        ema_model.load_state_dict(model.state_dict())
        ema_model.to(device)
        ema_model.eval()
        ema_updater = hydra.utils.instantiate(cfg.ema, model=ema_model)

    return model, ema_model, ema_updater


def build_optimizer_and_scheduler(cfg, model, batches_per_epoch):
    optimizer = model.configure_optimizer(**cfg.optimizer)

    grad_accum_steps = cfg.training.loop.grad_accum_steps
    optimizer_steps_per_epoch = (batches_per_epoch + grad_accum_steps - 1) // grad_accum_steps
    total_steps = optimizer_steps_per_epoch * cfg.training.loop.num_epochs

    scheduler = get_scheduler(
        optimizer=optimizer,
        name=cfg.training.lr_scheduler,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=total_steps,
    )

    return optimizer, scheduler


def _check_zarr_path_exists(zarr_path, key="dataset.zarr_path"):
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(
            f"{key}='{zarr_path}' does not exist. "
            f"Check that the Zarr dataset is available at the specified path."
        )


def _validate_augmentation_consistency(cfg):
    agent_cfg = cfg.agent
    pc_dim = agent_cfg.get('pc_dim')
    if pc_dim is None or pc_dim >= 6:
        return

    aug_cfg = cfg.dataset.get('augmentation_cfg')
    if aug_cfg is None:
        return

    pc_color = aug_cfg.get('pc', {}).get('color')
    if pc_color is not None:
        assert pc_dim >= 6, (
            f"PC color augmentation requires agent.pc_dim >= 6, got {pc_dim}. "
            f"The encoder only reads the first {pc_dim} channels (XYZ), "
            f"while color augmentation modifies channels 3:6 (RGB). "
            f"Either set agent.pc_dim=6 or remove augmentation_cfg.pc.color"
        )


def validate_config(cfg):
    # 验证窗口大小关系
    if cfg.n_obs_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps ({cfg.n_obs_steps}) cannot exceed horizon ({cfg.horizon})"
        )
    if cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_action_steps ({cfg.n_action_steps}) cannot exceed horizon ({cfg.horizon})"
        )

    if cfg.n_obs_steps - 1 + cfg.n_action_steps > cfg.horizon:
        raise ValueError(
            f"n_obs_steps-1 + n_action_steps ({cfg.n_obs_steps - 1 + cfg.n_action_steps}) "
            f"exceeds horizon ({cfg.horizon}). The control_action slice "
            f"pred[:, {cfg.n_obs_steps - 1}:{cfg.n_obs_steps - 1 + cfg.n_action_steps}] "
            f"would extend beyond the horizon dimension."
        )

    # 验证数据集路径存在性
    if cfg.dataset.get('zarr_path') is not None:
        _check_zarr_path_exists(cfg.dataset.zarr_path)
    elif cfg.dataset.get('datasets') is not None:
        for i, ds in enumerate(cfg.dataset.datasets):
            if ds.get('zarr_path') is not None:
                _check_zarr_path_exists(ds.zarr_path, f"dataset.datasets[{i}].zarr_path")

    # 验证优化器参数
    if cfg.optimizer.get('obs_lr') is not None:
        assert cfg.optimizer.obs_lr >= 0, "optimizer.obs_lr must be non-negative (0 means freeze)"

    # 验证 MultiTask 配置
    if cfg.dataset.get('task_texts') is not None:
        if len(cfg.dataset.task_texts) != len(cfg.dataset.task_names):
            raise ValueError(
                f"dataset.task_texts length ({len(cfg.dataset.task_texts)}) "
                f"must match dataset.task_names length ({len(cfg.dataset.task_names)})"
            )
        if cfg.dataset.get('task_weights') is not None:
            if len(cfg.dataset.task_weights) != len(cfg.dataset.task_names):
                raise ValueError(
                    f"dataset.task_weights length ({len(cfg.dataset.task_weights)}) "
                    f"must match dataset.task_names length ({len(cfg.dataset.task_names)})"
                )
        if cfg.get("env_runner") is not None and "task_configs" in cfg.env_runner:
            env_tasks = [t["task_name"] for t in cfg.env_runner.task_configs]
            if len(env_tasks) != len(cfg.dataset.task_names):
                raise ValueError(
                    f"env_runner.task_configs ({len(env_tasks)} tasks) "
                    f"must match dataset.task_names ({len(cfg.dataset.task_names)} tasks)"
                )
            env_texts = [t.get("task_text", t["task_name"]) for t in cfg.env_runner.task_configs]
            unknown = set(env_texts) - set(cfg.dataset.task_texts)
            if unknown:
                raise ValueError(
                    f"env_runner task_texts {unknown} not found in dataset.task_texts. "
                    f"Add them to dataset.task_texts for CLIP embedding cache coverage."
                )
            for task_cfg in cfg.env_runner.task_configs:
                task_name = task_cfg["task_name"]
                task_text = task_cfg.get("task_text", task_name)
                if task_name in cfg.dataset.task_names:
                    idx = cfg.dataset.task_names.index(task_name)
                    if task_text != cfg.dataset.task_texts[idx]:
                        raise ValueError(
                            f"env_runner task '{task_name}' uses task_text='{task_text}' "
                            f"but dataset.task_texts[{idx}]='{cfg.dataset.task_texts[idx]}' "
                            f"for the same task_name. Ensure dataset.task_names and "
                            f"dataset.task_texts are aligned, or update env_runner.task_configs."
                        )

    agent_cfg = cfg.agent

    # 验证 ManiFlow 特定约束
    if 'flow_batch_ratio' in agent_cfg:
        flow_ratio = agent_cfg.get('flow_batch_ratio', 0)
        assert 0.0 < flow_ratio < 1.0, \
            f"flow_batch_ratio ({flow_ratio}) must be in (0, 1)"
        tts_mode = agent_cfg.get('target_t_sample_mode', 'relative')
        assert tts_mode in ('relative', 'absolute'), \
            f"target_t_sample_mode must be 'relative' or 'absolute', got '{tts_mode}'"

        if cfg.dataloader.batch_size < 2:
            raise ValueError(
                f"ManiFlow requires batch_size >= 2 for consistency training "
                f"(flow/consistency split needs at least 2 samples). "
                f"Got batch_size={cfg.dataloader.batch_size}."
            )

        use_ema = cfg.training.get('use_ema', False)
        use_ema_teacher = cfg.training.get('use_ema_teacher_for_consistency', False)
        if not use_ema or not use_ema_teacher:
            import warnings
            warnings.warn(
                "ManiFlow without EMA teacher for consistency: "
                "the model will only train on flow loss (target_t=0). "
                "Euler ODE inference at target_t=dt>0 will experience "
                "condition distribution shift. "
                "Set training.use_ema=true and "
                "training.use_ema_teacher_for_consistency=true.",
                UserWarning,
            )

    # 验证 MoE 特定约束
    if 'num_experts' in agent_cfg:
        num_experts = agent_cfg.get('num_experts', 0)
        top_k = agent_cfg.get('top_k', 0)
        assert top_k <= num_experts, \
            f"top_k ({top_k}) must be <= num_experts ({num_experts})"

    # 验证数据增强与 encoder 的交叉一致性
    _validate_augmentation_consistency(cfg)

    # 验证 modality_dropout 配置
    if hasattr(cfg.agent, 'modality_dropout_probs') and cfg.agent.modality_dropout_probs:
        for modality, prob in cfg.agent.modality_dropout_probs.items():
            if prob > 0 and modality == 'action':
                import warnings
                warnings.warn(
                    f"modality_dropout for 'action' (prob={prob}) will NOT take effect: "
                    f"'action' does not flow through preprocess(). "
                    f"Use dataset augmentation for action noise instead.",
                    UserWarning,
                )

    # 校验 persistent_workers 与 epoch-dependent dataset 的兼容性
    if (cfg.dataloader.get('persistent_workers', False)
            and cfg.get('dataset', {}).get('sampling_strategy') is not None):
        raise ValueError(
            "persistent_workers=True is incompatible with MultiTaskDataset's "
            "epoch-dependent sampling. Set persistent_workers=False."
        )

    from termcolor import cprint
    cprint("✓ Config validation passed", "green")


def build_train_components(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script requires GPU.")

    device = torch.device(cfg.training.device)

    # 构建 dataset 和 normalizer
    dataset, normalizer = build_dataset_and_normalizer(cfg)

    # 构建 dataloader
    train_loader = DataLoader(dataset, worker_init_fn=worker_init_fn, **cfg.dataloader)
    val_dataset = dataset.get_validation_dataset()
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            worker_init_fn=worker_init_fn,
            **cfg.val_dataloader,
        )

    # 构建 model 和 EMA
    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)

    # 构建 optimizer 和 scheduler
    batches_per_epoch = len(train_loader)
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, batches_per_epoch)

    # 构建 workspace 和 env_runner
    workspace = hydra.utils.instantiate(cfg.workspace)

    env_runner = None
    if cfg.training.loop.eval_interval_epochs > 0 and cfg.get("env_runner") is not None:
        env_runner = hydra.utils.instantiate(cfg.env_runner)

    return {
        'device': device,
        'model': model,
        'ema_model': ema_model,
        'ema_updater': ema_updater,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'workspace': workspace,
        'env_runner': env_runner,
    }


@hydra.main(version_base=None, config_path="configs")
def main(cfg):
    validate_config(cfg)

    set_seed(cfg.training.seed)
    components = build_train_components(cfg)
    components['workspace'].save_hydra_config(cfg)

    trainer = Trainer(
        device=components['device'],
        model=components['model'],
        ema_model=components['ema_model'],
        ema_updater=components['ema_updater'],
        optimizer=components['optimizer'],
        scheduler=components['scheduler'],
        train_loader=components['train_loader'],
        val_loader=components['val_loader'],
        env_runner=components['env_runner'],
        workspace=components['workspace'],
        train_loop_cfg=cfg.training.loop,
        use_ema_teacher_for_consistency=cfg.training.use_ema_teacher_for_consistency,
    )
    trainer.train(resume_tag="latest")


if __name__ == "__main__":
    main()