from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import tempfile
import traceback

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
os.chdir(ROOT_DIR)

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dexmani_policy.common.checkpoint_io import CheckpointStore, TrainCheckpoint
from dexmani_policy.common.config import register_resolvers
from dexmani_policy.common.pytorch_util import (
    dict_apply,
    fix_state_dict,
    get_rng_state,
    set_seed,
)
from dexmani_policy.training.build_utils import (
    build_dataset_and_normalizer,
    build_model_and_ema,
    build_optimizer_and_scheduler,
    validate_config,
)
from dexmani_policy.training.resume import build_resume_contract, build_train_loader

register_resolvers()


def load_config(config_name: str):
    try:
        GlobalHydra.instance().clear()
    except (AttributeError, RuntimeError):
        pass
    config_dir = os.path.join(ROOT_DIR, "dexmani_policy", "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        # compose has no Hydra runtime; provide the only runtime value used by configs.
        cfg.workspace.output_dir = "/tmp/smoke_test_output"
        OmegaConf.resolve(cfg)
    return cfg


def _iter_targets(node, path: str = "$"):
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)

    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str):
            yield path, target
        for key, value in node.items():
            yield from _iter_targets(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_targets(value, f"{path}[{index}]")


def _split_target(target: str) -> tuple[str, str]:
    module_name, separator, attribute = target.rpartition(".")
    if not separator:
        raise ImportError(f"Invalid Hydra target: {target!r}")
    return module_name, attribute


def _validate_target_references(cfg) -> int:
    """Check target modules cheaply; import only the policy Agent target.

    Some runtime targets (notably env runners) intentionally import optional
    external packages such as dexmani_sim. Config-only validation should not
    require those runtime dependencies just to prove the policy config is
    structurally wired.
    """
    targets = list(_iter_targets(cfg))
    agent_target = cfg.get("agent", {}).get("_target_")

    for path, target in targets:
        module_name, attribute = _split_target(target)
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                f"Failed to locate Hydra target module {module_name!r} "
                f"for {target!r} at {path}"
            ) from exc
        if spec is None:
            raise ImportError(
                f"Failed to locate Hydra target module {module_name!r} "
                f"for {target!r} at {path}"
            )

        if target == agent_target:
            try:
                module = importlib.import_module(module_name)
                getattr(module, attribute)
            except Exception as exc:
                raise ImportError(
                    f"Failed to import agent target {target!r} at {path}"
                ) from exc

    return len(targets)


def validate_config_only(config_name: str):
    print(f"\n{'=' * 60}")
    print(f"Config check: {config_name}")
    print(f"{'=' * 60}")

    cfg = load_config(config_name)
    validate_config(cfg)

    target_count = _validate_target_references(cfg)

    print(f"      checked Hydra targets: {target_count}")
    print(f"\n✓ {config_name} config check PASSED\n")
    return True


def _prepare_dqrise_codebook(cfg, normalizer) -> str | None:
    """Create a schema-valid DQ-RISE codebook matching the smoke dataset normalizer."""
    agent_target = str(cfg.get("agent", {}).get("_target_", ""))
    if not agent_target.endswith(".DQRISEAgent"):
        return None

    from dexmani_policy.agents.vq_hand.codebook_manager import CodebookManager

    action_dim = int(cfg.agent.action_dim)
    tcp_dim = int(cfg.agent.tcp_dim)
    hand_dim = action_dim - tcp_dim
    num_groups = int(cfg.agent.get("codebook_num_groups", 2))
    codebook_size = int(cfg.agent.get("codebook_size", 4))
    total_codes = codebook_size**num_groups

    manager = CodebookManager(
        hand_dim=hand_dim,
        num_groups=num_groups,
        codebook_size=codebook_size,
    )

    # Deterministic synthetic prototypes are enough to exercise DQ-RISE's
    # integration path. Persist them through CodebookManager.save() so the
    # fixture always follows the runtime artifact schema instead of duplicating it.
    positions = torch.linspace(-1.0, 1.0, total_codes, dtype=torch.float32).unsqueeze(1)
    per_dim_offset = torch.linspace(
        -0.05, 0.05, hand_dim, dtype=torch.float32
    ).unsqueeze(0)
    normalized_poses = (positions + per_dim_offset).clamp(-1.0, 1.0)
    manager.sorted_hand_poses = (
        (normalized_poses + 1.0)
        * 0.5
        * (manager.hand_max - manager.hand_min)
        + manager.hand_min
    )
    manager.pca_permutation = torch.arange(total_codes, dtype=torch.long)
    manager.layer_weights = torch.full(
        (num_groups,), 1.0 / num_groups, dtype=torch.float32
    )

    action_params = normalizer["action"].params_dict
    manager.set_hand_normalizer(
        action_params["scale"][-hand_dim:],
        action_params["offset"][-hand_dim:],
    )

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as file:
        tmp_path = file.name
    manager.save(tmp_path)
    return tmp_path


def smoke_test(config_name: str):
    print(f"\n{'=' * 60}")
    print(f"Smoke test: {config_name}")
    print(f"{'=' * 60}")

    cfg = load_config(config_name)
    validate_config(cfg)
    set_seed(cfg.training.seed)

    print("[1/6] Building dataset & normalizer ...")
    dataset, normalizer = build_dataset_and_normalizer(cfg)
    train_loader = build_train_loader(cfg, dataset)
    print(f"      dataset size: {len(dataset)}, batches/epoch: {len(train_loader)}")

    val_dataset = dataset.get_validation_dataset()
    if val_dataset is not None:
        print(f"      val dataset size: {len(val_dataset)}")
        if (
            hasattr(dataset, "sampling_strategy")
            and dataset.sampling_strategy == "weighted"
        ):
            assert (
                hasattr(val_dataset, "task_weights")
                and val_dataset.task_weights is not None
            ), "MultiTaskDataset validation set must preserve task_weights for weighted strategy"
            print(
                "      ✓ weighted strategy validation set OK "
                f"(task_weights={val_dataset.task_weights})"
            )
    else:
        print("      no validation set (val_ratio=0)")

    codebook_tmp = _prepare_dqrise_codebook(cfg, normalizer)
    if codebook_tmp is not None:
        cfg.agent.codebook_path = codebook_tmp
        print(f"      [dqrise] temporary codebook → {codebook_tmp}")

    print("[2/6] Building model & EMA ...")
    device = torch.device(cfg.training.device)
    try:
        model, ema_model, ema_updater = build_model_and_ema(cfg, device, normalizer)
    finally:
        if codebook_tmp is not None:
            pathlib.Path(codebook_tmp).unlink(missing_ok=True)

    print("[3/6] Building optimizer & scheduler ...")
    optimizer, scheduler = build_optimizer_and_scheduler(cfg, model, len(train_loader))

    print("[4/6] Running forward + backward ...")
    batch = next(iter(train_loader))
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

    loss_kwargs = model.get_training_loss_kwargs(ema_model)
    model.train()
    raw_loss, loss_dict = model.compute_loss(batch, **loss_kwargs)
    raw_loss.backward()

    assert torch.isfinite(raw_loss), f"Non-finite loss: {raw_loss.item()}"
    print(f"      loss: {raw_loss.item():.4f}  keys: {list(loss_dict.keys())}")

    unreached = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if unreached:
        print(
            f"      ⚠ {len(unreached)} trainable params received no gradient in this batch "
            "(verify static usage across DDP batches):"
        )
        for name in unreached[:10]:
            print(f"        - {name}")
        if len(unreached) > 10:
            print(f"        ... and {len(unreached) - 10} more")
    else:
        print("      ✓ all trainable params received gradients")

    print("[5/6] Running predict_action ...")
    model.eval()
    with torch.no_grad():
        obs_sample = {key: value[:1] for key, value in batch["obs"].items()}
        result = model.predict_action(obs_sample)
        pred_shape = tuple(result["pred_action"].shape)
        ctrl_shape = tuple(result["control_action"].shape)
        expected_ctrl_dim = model.control_action_dim
        assert ctrl_shape == (
            1,
            cfg.n_action_steps,
            expected_ctrl_dim,
        ), (
            f"control_action shape {ctrl_shape} != "
            f"(1, {cfg.n_action_steps}, {expected_ctrl_dim})"
        )
        print(f"      pred_action: {pred_shape}  control_action: {ctrl_shape}")

    print("[6/6] Checkpoint save → load roundtrip ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = pathlib.Path(tmpdir)
        store = CheckpointStore(ckpt_dir)

        model_sd = {key: value.clone() for key, value in model.state_dict().items()}
        ema_sd = (
            {key: value.clone() for key, value in ema_model.state_dict().items()}
            if ema_model is not None
            else None
        )

        ckpt = TrainCheckpoint(
            epoch=0,
            global_step=1,
            next_micro_step=0,
            model_state=fix_state_dict(model_sd, is_current_ddp=False),
            ema_model_state=(
                fix_state_dict(ema_sd, is_current_ddp=False)
                if ema_sd is not None
                else None
            ),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            monitor={"test_mean_score": 0.85},
            resume_contract=build_resume_contract(cfg, model, train_loader),
            ema_updater_step=None,
            ema_decay=None,
            rng_states=[get_rng_state()],
        )
        ckpt_path = store.save(
            "epoch=0000-step=00000001-score=0.8500.pt",
            ckpt,
        )
        print(f"      saved checkpoint: {ckpt_path.name}")

        loaded = store.load(ckpt_path)
        assert loaded.epoch == 0
        assert loaded.global_step == 1
        assert loaded.next_micro_step == 0
        assert loaded.monitor.get("test_mean_score") == 0.85
        assert (
            loaded.resume_contract["training"]["num_training_steps"]
            == cfg.training.loop.total_train_steps
        )

        loaded_model_sd = fix_state_dict(loaded.model_state, is_current_ddp=False)
        loaded_model_sd = {
            key: value.to(device) for key, value in loaded_model_sd.items()
        }
        for key in model_sd:
            if not torch.equal(model_sd[key], loaded_model_sd[key]):
                raise AssertionError(
                    f"Model state dict mismatch for key '{key}' after roundtrip"
                )
        print("      ✓ model state dict roundtrip OK")

        if ema_sd is not None and loaded.ema_model_state is not None:
            loaded_ema_sd = fix_state_dict(
                loaded.ema_model_state, is_current_ddp=False
            )
            loaded_ema_sd = {
                key: value.to(device) for key, value in loaded_ema_sd.items()
            }
            for key in ema_sd:
                if not torch.equal(ema_sd[key], loaded_ema_sd[key]):
                    raise AssertionError(
                        f"EMA state dict mismatch for key '{key}' after roundtrip"
                    )
            print("      ✓ EMA state dict roundtrip OK")

        agent_contract = loaded.resume_contract["agent"]
        assert agent_contract["n_obs_steps"] == model.n_obs_steps
        assert agent_contract["n_action_steps"] == model.n_action_steps
        assert agent_contract["action_dim"] == model.action_dim
        print("      ✓ resume contract roundtrip OK")

    print(f"\n✓ {config_name} smoke test PASSED\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate one or more DexMani Hydra policy configs."
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "Resolve/validate config, check Hydra target modules, and import the "
            "agent target without data or GPU execution."
        ),
    )
    parser.add_argument("config_names", nargs="+", help="Hydra config names to validate.")
    args = parser.parse_args()

    runner = validate_config_only if args.config_only else smoke_test
    for config_name in args.config_names:
        try:
            runner(config_name)
        except Exception as exc:
            mode = "config check" if args.config_only else "smoke test"
            print(f"\n✗ {config_name} {mode} FAILED: {exc}\n")
            traceback.print_exc()
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
