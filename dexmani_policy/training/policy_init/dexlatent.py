from .common import fit_normalizers, instantiate_policy


def build_fresh_policy(agent, initialization, normalization, dataset, field_dims):
    model = instantiate_policy(agent, dataset)
    model.obs_encoder.rgb_encoder.initialize_weights(initialization.rgb)
    hand_affine = model.hand_prior.initialize_weights(initialization.dexlatent)
    fit_normalizers(model, dataset, normalization, field_dims, hand_affine=hand_affine)
    latents = (
        model._encode_action_parts(chunk)[1]
        for chunk in dataset.iter_normalization_data("action")
    )
    model.normalizer.fit_field("hand_latent", latents)
    return model
