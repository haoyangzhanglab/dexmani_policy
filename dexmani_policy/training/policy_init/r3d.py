from .common import fit_normalizers, instantiate_policy


def build_fresh_policy(agent, initialization, normalization, dataset, field_dims):
    model = instantiate_policy(agent, dataset)
    model.obs_encoder.pc_encoder.initialize_weights(initialization.uni3d)
    fit_normalizers(model, dataset, normalization, field_dims)
    return model
