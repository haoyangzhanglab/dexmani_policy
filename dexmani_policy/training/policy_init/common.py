import hydra

from dexmani_policy.common.normalizer import normalization_feature_shapes
from dexmani_policy.training.config import validate_normalization
from dexmani_policy.training.data import fit_action_normalizer, fit_observation_normalizer


def instantiate_policy(agent, dataset):
    model = hydra.utils.instantiate(agent, _convert_="all")
    if dataset.episode_store[dataset.action_key].shape[-1] != model.control_action_dim:
        raise ValueError("Dataset primary action width must match policy control output")
    return model


def configure_normalizer(model, normalization, field_dims):
    validate_normalization(model, normalization, field_dims)
    feature_shapes = normalization_feature_shapes(field_dims, normalization)
    model.configure_normalizer(field_dims, feature_shapes)
    return feature_shapes


def fit_normalizers(
    model, dataset, normalization, field_dims, *,
    pretrained_state=None, inherited_fields=(), hand_affine=None,
):
    feature_shapes = configure_normalizer(model, normalization, field_dims)
    observation_fields = [key for key in feature_shapes if key != "action"]
    fit_observation_normalizer(
        dataset, model.normalizer, observation_fields, initial_state=pretrained_state,
        inherited_fields=inherited_fields,
    )
    fit_action_normalizer(dataset, model.normalizer, hand_affine=hand_affine)
    print(f"[Normalizer] train source frames; inherited fields={list(inherited_fields)}")
