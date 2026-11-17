from omegaconf import open_dict


def resolve_rgb_architecture(agent, source):
    name = agent.rgb_backbone_name
    if name not in ("dino", "siglip"):
        return
    rgb = dict(agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is None:
        if source is None:
            raise ValueError("Fresh HF RGB requires initialization.rgb or an explicit architecture")
        from transformers import AutoConfig, SiglipVisionConfig

        config_class = SiglipVisionConfig if name == "siglip" else AutoConfig
        rgb["architecture"] = config_class.from_pretrained(source).to_dict()
    with open_dict(agent):
        agent.rgb_backbone_config = rgb


def inherit_rgb_architecture(agent, saved_agent):
    if agent.rgb_backbone_name != saved_agent["rgb_backbone_name"]:
        raise ValueError("RGB backbone differs from initialization policy")
    rgb = dict(agent.get("rgb_backbone_config") or {})
    if rgb.get("architecture") is None:
        architecture = saved_agent["rgb_backbone_config"].get("architecture")
        if architecture is not None:
            rgb["architecture"] = architecture
    with open_dict(agent):
        agent.rgb_backbone_config = rgb
