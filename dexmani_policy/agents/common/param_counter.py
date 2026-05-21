from termcolor import cprint


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def print_param_count(agent):
    total, _ = count_params(agent)

    cprint(f'[{type(agent).__name__}] Parameter Count', 'cyan', attrs=['bold'])
    cprint(f'  Total: {total / 1e6:.2f} M', 'white')

    for name, child in agent.named_children():
        t, tr = count_params(child)
        frozen = t - tr
        color = 'green' if tr > 0 else 'white'
        cprint(f'  {name:<20}: {t / 1e6:.2f} M  (trainable={tr / 1e6:.2f} M  frozen={frozen / 1e6:.2f} M)', color)
