import torch
import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.empty(()))

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype