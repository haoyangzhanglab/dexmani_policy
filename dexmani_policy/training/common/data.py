import hydra
import torch
from torch.utils.data import DataLoader

from dexmani_policy.agents.policies.base import BasePolicy
from dexmani_policy.common.config import resolve_normalization_spec, FINGER_VECTOR_FIELDS, normalization_feature_shapes
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.datasets.hand_prior.metadata import describe_hand_data
from dexmani_policy.datasets.hand_prior.latent_flow_dataset import LatentFlowDataset
from dexmani_policy.deployment.metadata import describe_real_runtime


def build_dataset_and_normalizer(cfg, *, resume_checkpoint=None, verbose=True):
    """Construct one admitted train/val source; resume only restores saved affine."""
    if cfg.get("resume_from") is not None and resume_checkpoint is None:
        raise ValueError("Resume requires the checkpoint loaded by the training entry point")
    if resume_checkpoint is not None:
        from dexmani_policy.training.common.resume import validate_resume_protocol
        validate_resume_protocol(resume_checkpoint.effective_config)
    spec = resolve_normalization_spec(cfg)
    agent_class: type[BasePolicy] = hydra.utils.get_class(cfg.agent._target_)
    if agent_class.requires_split_manifest and cfg.dataset.get("split_manifest") is None:
        raise ValueError("Hand pretraining requires an explicit shared split_manifest")
    dataset = hydra.utils.instantiate(cfg.dataset)
    buffer = dataset.replay_buffer
    dims = {key: int(buffer[key].shape[-1]) for key, mode in spec.items() if key != "action" and mode != "identity"}
    for key in FINGER_VECTOR_FIELDS.intersection(dims):
        dims[key] = 15
    if "point_cloud" in dims and dataset.pc_dim is not None:
        dims["point_cloud"] = dataset.pc_dim
    dims["action"] = dataset.action_dim
    if dims["action"] != cfg.agent.action_dim:
        raise ValueError("Dataset action width including auxiliary targets must match agent.action_dim")
    description = {"normalization_dims": dims, "split_manifest": dataset.split_manifest}
    runtime = describe_real_runtime(buffer, dataset.sensor_modalities, cfg.task_name)
    if runtime is not None:
        description["real_runtime"] = runtime
    if agent_class.requires_hand_pretraining:
        description["hand_data"] = dataset.hand_data if isinstance(dataset, LatentFlowDataset) else describe_hand_data(
            buffer, action_key=dataset.action_key, horizon=dataset.horizon,
            obs_horizon=dataset.obs_horizon, pad_before=dataset.pad_before, pad_after=dataset.pad_after,
            history_steps=getattr(dataset, "history_steps", None),
        )
    if resume_checkpoint is None:
        normalizer = dataset.get_normalizer(spec)
    else:
        normalizer = LinearNormalizer(normalization_feature_shapes({key: dims[key] for key, mode in spec.items() if mode != "identity"}))
        normalizer.load_state_dict({
            key.removeprefix("normalizer."): value
            for key, value in resume_checkpoint.model_state.items() if key.startswith("normalizer.")
        }, strict=True)
        dataset._normalizer = normalizer
    if verbose:
        print(f"[Dataset] {dataset.training_summary()}")
        if resume_checkpoint is None:
            print("[Normalizer] all real train-episode frames; relative actions retain each window/reference")
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
