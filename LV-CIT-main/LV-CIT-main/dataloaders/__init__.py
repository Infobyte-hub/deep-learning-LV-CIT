from .lvcit_voc import LvcitVoc, LvcitVoc2
from .lvcit_coco import LvcitCoco, LvcitCoco2
from .default_voc import Voc2007Classification
from .default_coco import COCO2014Classification
from .default_voc2 import Voc2007Classification2
from .default_coco2 import COCO2014Classification2

__all__ = [
    'LvcitVoc',
    'LvcitVoc2',
    'LvcitCoco',
    'LvcitCoco2',

    'Voc2007Classification',
    'COCO2014Classification',
    'Voc2007Classification2',
    'COCO2014Classification2',
]
