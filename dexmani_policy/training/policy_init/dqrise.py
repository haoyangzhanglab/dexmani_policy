from omegaconf import open_dict

from dexmani_policy.agents.hand_prior.codebook import load_codebook

from .common import fit_normalizers, instantiate_policy
from .rgb import inherit_rgb_architecture


def build_fresh_policy(agent, initialization, normalization, dataset, field_dims):
    if initialization.codebook is None:
        raise ValueError("Fresh DQ-RISE training requires initialization.codebook")
    codebook, hand_affine = load_codebook(initialization.codebook)
    with open_dict(agent):
        agent.num_codes = codebook.num_codes
    model = instantiate_policy(agent, dataset)
    model.obs_encoder.rgb_encoder.initialize_weights(initialization.rgb)
    model.hand_codebook = codebook
    fit_normalizers(model, dataset, normalization, field_dims, hand_affine=hand_affine)
    return model


def prepare_warm_start(agent, saved_agent):
    inherit_rgb_architecture(agent, saved_agent)
    if agent.get("num_codes") is None:
        with open_dict(agent):
            agent.num_codes = saved_agent["num_codes"]
