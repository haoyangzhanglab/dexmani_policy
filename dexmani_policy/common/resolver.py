import warnings
from omegaconf import OmegaConf

def register_resolvers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"__builtins__": {}}, {}), replace=True)
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b, replace=True)
