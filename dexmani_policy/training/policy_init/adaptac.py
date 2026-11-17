def prepare_warm_start(agent, saved_agent):
    if agent.tactile_encoder._target_ != saved_agent["tactile_encoder"]["_target_"]:
        raise ValueError("Tactile encoder differs from initialization policy")
