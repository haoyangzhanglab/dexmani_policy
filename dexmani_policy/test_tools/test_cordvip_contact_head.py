import torch

from dexmani_policy.training.pretrain.obs.cordvip import CordViP


def test_cordvip_contact_follows_object_point_order():
    generator = torch.Generator().manual_seed(14)
    normalization = dict.fromkeys(("hand_pointcloud", "object_pointcloud", "joint_state"), "identity")
    model = CordViP(
        normalization, n_obs_steps=1, arm_loss_weight=0, hand_loss_weight=0,
    )
    obs = {
        "hand_pointcloud": torch.rand(1, 1, 64, 3, generator=generator),
        "object_pointcloud": torch.rand(1, 1, 1024, 3, generator=generator),
        "joint_state": torch.zeros(1, 1, 19),
    }
    labels = obs["object_pointcloud"][..., 0]
    permutation = torch.randperm(1024, generator=generator)
    shuffled = {**obs, "object_pointcloud": obs["object_pointcloud"][..., permutation, :]}

    predictions = []
    with (
        torch.no_grad(),
        model.contact_head.register_forward_hook(lambda _module, _inputs, output: predictions.append(output)),
    ):
        loss, _ = model({"obs": obs, "contact_map": labels})
        shuffled_loss, _ = model({"obs": shuffled, "contact_map": labels[..., permutation]})

    torch.testing.assert_close(predictions[1], predictions[0][..., permutation])
    torch.testing.assert_close(shuffled_loss, loss)
    assert predictions[0].std() > 0
