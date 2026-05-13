from .rgb import RGBAug
from .pc_color import PCColorJitter
from .pc_spatial import PCSpatialAug
from .pc_dropout import PCDropout
from .state import StateNoiseAug

PC_AUG_CLASSES = {
    'color': PCColorJitter,
    'spatial': PCSpatialAug,
    'dropout': PCDropout,
}

__all__ = [
    'RGBAug',
    'PCColorJitter', 'PCSpatialAug', 'PCDropout',
    'StateNoiseAug',
    'PC_AUG_CLASSES',
]
