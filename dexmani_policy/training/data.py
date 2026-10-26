import copy

import hydra
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import resolve_normalization_spec
from dexmani_policy.common.normalization import AffineNormalizer


def build_normalizer(dataset, spec: dict, action_key: str) -> AffineNormalizer:
    """Fit consumed training data, preserving rot6d coordinates for action_ee auto mode."""
    normalizer = AffineNormalizer()

    for key, mode in spec.items():
        if mode == "identity":
            continue

        mixed_action = key == "action" and mode == "auto" and action_key == "action_ee"
        mode = "limits" if mode == "auto" else mode
        normalizer.fit_field(key, dataset.iter_normalization_data(key), mode=mode)
        if mixed_action:
            field = normalizer[key]
            if field.params_dict["scale"].numel() <= 9:
                raise ValueError("action_ee requires xyz, rot6d and hand coordinates")
            field.params_dict["scale"][3:9] = 1
            field.params_dict["offset"][3:9] = 0
            for name, value in (("min", -1), ("max", 1), ("mean", 0), ("std", 1)):
                field.input_stats[name][3:9] = value

    return normalizer


def build_dataset_and_normalizer(cfg, *, resume_checkpoint=None, verbose=True):
    """Build data and fit fresh statistics; full resume restores the model's normalizer."""
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        raise ValueError("Resume requires the checkpoint loaded by the training entry point")
    spec = resolve_normalization_spec(cfg)
    dataset_cfg = copy.deepcopy(cfg.dataset)
    saved_recipes = (
        resume_checkpoint.effective_config["data_recipe"] if resume_checkpoint is not None else cfg.get("data_recipe")
    )
    if saved_recipes is not None:
        with open_dict(dataset_cfg):
            dataset_cfg.split_manifest = saved_recipes[0]["split_manifest"]
    agent_class: type[BasePolicy] = hydra.utils.get_class(cfg.agent._target_)
    if agent_class.requires_split_manifest and dataset_cfg.get("split_manifest") is None:
        raise ValueError("Hand pretraining requires an explicit shared split_manifest")
    dataset = hydra.utils.instantiate(dataset_cfg)
    description = dataset.describe_training_data(
        spec, task_name=cfg.task_name,
        requires_hand_pretraining=agent_class.requires_hand_pretraining,
    )
    if description["normalization_dims"]["action"] != cfg.agent.action_dim:
        raise ValueError("Dataset action width including auxiliary targets must match agent.action_dim")
    normalizer = None
    if resume_checkpoint is None:
        normalizer = build_normalizer(dataset, spec, cfg.action_key)
    if verbose:
        print(f"[Dataset] {dataset.training_summary()}")
        if resume_checkpoint is None:
            print("[Normalizer] consumed training rows; relative actions count each window/reference")
    return dataset, normalizer, description


def close_data_loader(loader, iterator=None):
    # Persistent workers survive early exits until their iterator is shut down.
    iterator = loader._iterator if iterator is None else iterator
    loader._iterator = None
    if iterator is not None and loader.num_workers > 0:
        iterator._shutdown_workers()


def build_train_loader(cfg, dataset):
    options = dict(cfg.dataloader)
    if options.get("num_workers", 0) == 0:
        options["persistent_workers"] = False
        options.pop("prefetch_factor", None)
    samples = len(dataset)
    batch_size = options["batch_size"]
    accumulation = cfg.training.loop.gradient_accumulation_steps
    batches = (samples + batch_size - 1) // batch_size
    if (
        accumulation > 1
        and not options.get("drop_last", False)
        and samples % batch_size
        and batches % accumulation != 1
    ):
        raise ValueError(
            f"Mixed micro-batch sizes in accumulation group: "
            f"N={samples}, B={batch_size}, A={accumulation}; "
            "use drop_last=True or gradient_accumulation_steps=1"
        )
    generator = torch.Generator().manual_seed(cfg.training.seed)
    loader = DataLoader(dataset, generator=generator, **options)
    if len(loader) == 0:
        raise ValueError(
            "Training DataLoader is empty; check dataset length, batch_size and drop_last"
        )
    return loader
