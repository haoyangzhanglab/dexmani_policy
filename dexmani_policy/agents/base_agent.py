import torch
from typing import Dict
from dexmani_policy.common.normalizer import LinearNormalizer
from dexmani_policy.agents.common.module_attr_mixin import ModuleAttrMixin


class BaseAgent(ModuleAttrMixin):

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def obs_data_process(self, obs_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError

    def get_optimizer(self, lr: float, weight_decay: float, **kwargs):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
        return optimizer

    def compute_loss(self, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def predict_action(self):
        raise NotImplementedError