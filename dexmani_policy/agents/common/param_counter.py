from termcolor import cprint


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_param_count(agent):
    total, _                   = count_params(agent)
    enc_total, enc_trainable   = count_params(agent.obs_encoder)
    dec_total, _               = count_params(agent.action_decoder)
    enc_frozen                 = enc_total - enc_trainable

    cprint(f'[{type(agent).__name__}] Parameter Count', 'cyan', attrs=['bold'])
    cprint(f'  Total         : {total         / 1e6:.2f} M', 'white')
    cprint(f'  ObsEncoder    : {enc_total      / 1e6:.2f} M'
           f'  (trainable={enc_trainable / 1e6:.2f} M  frozen={enc_frozen / 1e6:.2f} M)', 'green')
    cprint(f'  ActionDecoder : {dec_total      / 1e6:.2f} M', 'yellow')
