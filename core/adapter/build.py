from .base_adapter import BaseAdapter
from .rotta import RoTTA
from .tribe import TRIBE
from .tribe_bbn import TRIBE_BBN

from .lame import LAME
from .tent import TENT
from .bn import BN
from .note import NOTE
from .cotta import CoTTA
from .test import TEST
from .unitta_cofa import UNITTA_COFA
from .unitta_bdn import UNITTA_BDN
from .unitta import UNITTA
from .roid import ROID
from .unmixtns import UNMIXTNS


def build_adapter(cfg) -> BaseAdapter:
    if cfg.ADAPTER.NAME == "rotta":
        return RoTTA
    elif cfg.ADAPTER.NAME == "tribe":
        return TRIBE
    elif cfg.ADAPTER.NAME == "tribe_bbn":
        return TRIBE_BBN
    elif cfg.ADAPTER.NAME == "lame":
        return LAME
    elif cfg.ADAPTER.NAME == "tent":
        return TENT
    elif cfg.ADAPTER.NAME == "bn":
        return BN
    elif cfg.ADAPTER.NAME == "note":
        return NOTE
    elif cfg.ADAPTER.NAME == "cotta":
        return CoTTA
    elif cfg.ADAPTER.NAME == "petal":
        return PETALFim
    elif cfg.ADAPTER.NAME == "test":
        return TEST
    elif cfg.ADAPTER.NAME == "unitta_cofa":
        return UNITTA_COFA
    elif cfg.ADAPTER.NAME == "unitta_bdn":
        return UNITTA_BDN
    elif cfg.ADAPTER.NAME == "unitta":
        return UNITTA
    elif cfg.ADAPTER.NAME == "roid":
        return ROID
    elif cfg.ADAPTER.NAME == "unmixtns":
        return UNMIXTNS

    else:
        raise NotImplementedError("Implement your own adapter")
