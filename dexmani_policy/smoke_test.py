import os
import pathlib
import sys
import tempfile

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dexmani_policy.common.pytorch_util import set_seed, worker_init_fn, dict_apply, fix_state_dict
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.training.common.checkpoint_io import CheckpointStore, TrainCheckpoint
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
        # compose has no Hydra runtime — manually bypass ${hydra:runtime.output_dir}
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
    # loss_kwargs logic intentionally mirrors trainer.py; duplicated so smoke test stays self-contained
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

    # 5.1 MoE enhanced gate smoke check (exercises the use_enhanced_gate=True path
    # that is also covered by the __main__ tests in plugins/moe.py and core/moe.py)
    if 'num_experts' in cfg.agent and cfg.agent.get('use_enhanced_gate', False) is False:
        print("[5.1/6] MoE enhanced gate smoke check ...")
        from dexmani_policy.agents.obs_encoder.plugins.moe import MoE
        moe_gate = MoE(
            dim=64, num_experts=4, top_k=2,
            hidden_dim=64, out_dim=64, num_layers=1,
            use_enhanced_gate=True, gate_dropout=0.1,
        ).to(device)
        x_gate = torch.randn(8, 64, device=device)
        z_gate, aux_gate = moe_gate(x_gate, return_aux=True)
        assert torch.isfinite(z_gate).all(), "Enhanced gate output non-finite"
        assert torch.isfinite(aux_gate['loss']), "Enhanced gate aux loss non-finite"
        print(f"      ✓ enhanced gate: z={z_gate.shape}, aux_loss={aux_gate['loss'].item():.4f}")

    print("[6/6] Checkpoint save → load roundtrip ...")
    accum_steps = max(1, int(cfg.training.loop.get('gradient_accumulation_steps', 1)))
    num_training_steps = -(-len(train_loader) // accum_steps) * cfg.training.loop.num_epochs

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = pathlib.Path(tmpdir)
        store = CheckpointStore(ckpt_dir)

        # Capture pre-save state dicts (unwrap from compile if needed)
        model_sd = {k: v.clone() for k, v in model.state_dict().items()}
        ema_sd = {k: v.clone() for k, v in ema_model.state_dict().items()} if ema_model is not None else None
        opt_sd = optimizer.state_dict()
        sched_sd = scheduler.state_dict()

        # Build and save a checkpoint
        ckpt = TrainCheckpoint(
            epoch=0,
            global_step=1,
            model_state=model_sd,
            ema_model_state=fix_state_dict(ema_sd, is_current_ddp=False) if ema_sd is not None else None,
            optimizer_state=opt_sd,
            scheduler_state=sched_sd,
            monitor={"test_mean_score": 0.85},
            train_params={
                'n_obs_steps': model.n_obs_steps,
                'n_action_steps': model.n_action_steps,
                'action_dim': model.action_dim,
                'horizon': model.horizon,
                'action_key': getattr(model, 'action_key', 'action'),
                'num_training_steps': num_training_steps,
            },
        )
        ckpt_path = store.save("epoch=0000-step=00000001-score=0.8500.pt", ckpt)
        print(f"      saved checkpoint: {ckpt_path.name}")

        # Reload
        loaded = store.load(ckpt_path)
        assert loaded.epoch == 0
        assert loaded.global_step == 1
        assert loaded.monitor.get("test_mean_score") == 0.85
        assert loaded.train_params.get('num_training_steps') == num_training_steps

        # Verify model state dict roundtrip (loaded is on CPU, move to model device)
        loaded_model_sd = fix_state_dict(loaded.model_state, is_current_ddp=False)
        loaded_model_sd = {k: v.to(device) for k, v in loaded_model_sd.items()}
        for key in model_sd:
            if not torch.equal(model_sd[key], loaded_model_sd[key]):
                raise AssertionError(f"Model state dict mismatch for key '{key}' after roundtrip")
        print("      ✓ model state dict roundtrip OK")

        # Verify EMA state dict roundtrip
        if ema_sd is not None and loaded.ema_model_state is not None:
            loaded_ema_sd = fix_state_dict(loaded.ema_model_state, is_current_ddp=False)
            loaded_ema_sd = {k: v.to(device) for k, v in loaded_ema_sd.items()}
            for key in ema_sd:
                if not torch.equal(ema_sd[key], loaded_ema_sd[key]):
                    raise AssertionError(f"EMA state dict mismatch for key '{key}' after roundtrip")
            print("      ✓ EMA state dict roundtrip OK")

        # Verify train_params roundtrip
        tp = loaded.train_params
        assert tp.get('n_obs_steps') == model.n_obs_steps
        assert tp.get('n_action_steps') == model.n_action_steps
        assert tp.get('action_dim') == model.action_dim
        print("      ✓ train_params roundtrip OK")

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
