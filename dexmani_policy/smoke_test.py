import os
import pathlib
import sys

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.common.pytorch_util import set_seed, worker_init_fn, dict_apply
from dexmani_policy.common.resolver import register_resolvers
from dexmani_policy.train import build_dataset_and_normalizer, build_model_and_ema, build_optimizer_and_scheduler

register_resolvers()


def load_config(config_name: str):
    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass
    config_dir = os.path.join(ROOT_DIR, "dexmani_policy", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        # compose 无 Hydra runtime，手动绕过 ${hydra:runtime.output_dir}
        cfg.workspace.output_dir = "/tmp/smoke_test_output"
        OmegaConf.resolve(cfg)
    return cfg


def smoke_test(config_name: str):
    print(f"\n{'='*60}")
    print(f"Smoke test: {config_name}")
    print(f"{'='*60}")

    cfg = load_config(config_name)

    set_seed(cfg.training.seed)

    print("[1/5] Building dataset & normalizer ...")
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    train_loader = DataLoader(dataset, worker_init_fn=worker_init_fn, **cfg.dataloader)
    print(f"      dataset size: {len(dataset)}, batches/epoch: {len(train_loader)}")

    # 1.1 Validation dataset (regression test for weighted strategy bug)
    val_dataset = dataset.get_validation_dataset()
    if val_dataset is not None:
        print(f"      val dataset size: {len(val_dataset)}")
        if hasattr(dataset, 'sampling_strategy') and dataset.sampling_strategy == 'weighted':
            assert hasattr(val_dataset, 'task_weights') and val_dataset.task_weights is not None, \
                "MultiTaskDataset validation set must preserve task_weights for weighted strategy"
            print(f"      ✓ weighted strategy validation set OK (task_weights={val_dataset.task_weights})")
    else:
        print("      no validation set (val_ratio=0)")

    print("[2/5] Building model & EMA ...")
    device = torch.device(cfg.training.device)
    model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)

    print("[3/5] Building optimizer & scheduler ...")
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, len(train_loader))

    print("[4/5] Running forward + backward ...")
    batch = next(iter(train_loader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

    use_ema_teacher = cfg.training.use_ema_teacher_for_consistency
    # loss_kwargs 构造与 trainer.py:108 逻辑一致，有意重复以保证 smoke test 独立可运行
    loss_kwargs = {"ema_backbone": ema_model.action_decoder.model} if use_ema_teacher and ema_model else {}
    model.train()
    raw_loss, loss_dict = model.compute_loss(batch, **loss_kwargs)
    raw_loss.backward()

    assert torch.isfinite(raw_loss), f"Non-finite loss: {raw_loss.item()}"
    print(f"      loss: {raw_loss.item():.4f}  keys: {list(loss_dict.keys())}")

    print("[5/5] Running predict_action ...")
    model.eval()
    with torch.no_grad():
        obs_sample = {k: v[:1] for k, v in batch["obs"].items()}
        result = model.predict_action(obs_sample)
        pred_shape = tuple(result["pred_action"].shape)
        ctrl_shape = tuple(result["control_action"].shape)
        assert ctrl_shape == (1, cfg.n_action_steps, cfg.action_dim), \
            f"control_action shape {ctrl_shape} != (1, {cfg.n_action_steps}, {cfg.action_dim})"
        print(f"      pred_action: {pred_shape}  control_action: {ctrl_shape}")

    print(f"\n✓ {config_name} smoke test PASSED\n")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <config_name> [config_name ...]")
        print("Example: python smoke_test.py dp3")
        print("         python smoke_test.py dp3 maniflow")
        sys.exit(1)

    for name in sys.argv[1:]:
        try:
            smoke_test(name)
        except Exception as e:
            print(f"\n✗ {name} smoke test FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)
