import argparse
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from dexmani_policy.common.config import load_policy_config, positive_int
from dexmani_policy.common.io_util import atomic_path
from dexmani_policy.common.normalizer import LinearNormalizer, validate_normalizer_state
from dexmani_policy.common.pytorch_util import dict_apply, get_rng_state
from dexmani_policy.training.common.data import build_train_loader
from dexmani_policy.training.common.logging import accumulate_metrics
from dexmani_policy.training.common.resume import resolve_input_recipe


def parse_run(method):
    parser = argparse.ArgumentParser(description=f"Pretrain the {method} observation encoder")
    parser.add_argument("--config", default=f"pretrain/obs/{method}")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    if args.max_updates is not None:
        positive_int(args.max_updates, "max_updates")
    cfg = load_policy_config(args.config, args.overrides)
    output = args.output_dir or cast(dict, OmegaConf.to_container(cfg.workspace, resolve=False))["output_dir"]
    checkpoint = None
    if args.resume_from is not None:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
    with open_dict(cfg):
        cfg.resume_from = args.resume_from
    cfg = resolve_input_recipe(
        cfg, overrides=args.overrides,
        saved_config=None if checkpoint is None else checkpoint["effective_config"],
    )
    cfg.workspace.output_dir = output
    for name in ("encoder.pt", "resume.pt"):
        path = Path(cfg.workspace.output_dir) / name
        if path.exists():
            raise FileExistsError(f"Pretraining output already exists: {path}")
    return cfg, checkpoint, args.max_updates


def prepare_data(cfg, checkpoint):
    dataset = hydra.utils.instantiate(cfg.dataset)
    cfg.dataset.split_manifest = dataset.split_manifest
    if checkpoint is None:
        normalizer = dataset.get_normalizer(cfg.normalization)
    else:
        shapes = {key: (19,) if key == "joint_state" else (3,)
                  for key, mode in cfg.normalization.items() if mode != "identity"}
        normalizer = LinearNormalizer(shapes)
        normalizer.load_state_dict({
            key.removeprefix("normalizer."): value
            for key, value in checkpoint["model_state"].items() if key.startswith("normalizer.")
        }, strict=True)
        validate_normalizer_state(normalizer)
        dataset._normalizer = normalizer
    loader = build_train_loader(cfg, dataset)
    return dataset, normalizer, loader


def save_run(model, encoder, metadata, dataset, optimizer, scheduler, cfg, epoch, step):
    output = Path(cfg.workspace.output_dir)
    config = OmegaConf.to_container(cfg, resolve=True)
    payload = {
        **metadata,
        "encoder_state": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "normalizer_state": {key: value.detach().cpu() for key, value in model.normalizer.state_dict().items()},
        "normalization": dict(cfg.normalization),
        "pretrain_config": config,
        "split_manifest": dataset.split_manifest,
        "source": {"zarr_path": dataset.zarr_path,
                   "source_id": dataset.replay_buffer.attrs.get("source_id"),
                   "dt": dataset.replay_buffer.attrs.get("dt")},
        "epoch": epoch, "global_step": step,
    }
    with atomic_path(output / "encoder.pt") as temporary:
        torch.save(payload, temporary)
    checkpoint = {
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "epoch": epoch, "global_step": step,
        "rng_state": get_rng_state(cfg.training.device), "effective_config": config,
    }
    with atomic_path(output / "resume.pt") as temporary:
        torch.save(checkpoint, temporary)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        _, metrics = model(dict_apply(batch, lambda x: x.to(device)))
        accumulate_metrics(totals, metrics)
    return {f"val/{key}": value / len(loader) for key, value in totals.items()}
