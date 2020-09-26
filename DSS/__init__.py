from collections import namedtuple
import torch
import numpy as np
from .logger import get_logger
from collections import OrderedDict

logger_py = get_logger(__name__)

_debug = False
_debugging_tensor = None

def set_deterministic_():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

# Each attribute contains list of tensors or dictionaries, where
# each element in the list is a sample in the minibatch.
# If dictionaries are used, then the (keys, tensor) will be used to plot
# debugging visuals separately.
class DebuggingTensor:
    __slots__ = ['pts_world',
                 'pts_world_grad',
                 'img_mask_grad']

    def __init__(self,):
        self.pts_world = OrderedDict()
        self.pts_world_grad = OrderedDict()
        self.img_mask_grad = OrderedDict()


def set_debugging_mode_(is_debug, *args, **kwargs):
    global _debugging_tensor, _debug
    _debug = is_debug
    if _debug:
        _debugging_tensor = DebuggingTensor(*args, **kwargs)
        logger_py.info('Enabled debugging mode.')
    else:
        _debugging_tensor = None


def get_debugging_mode():
    return _debug


def get_debugging_tensor():
    if _debugging_tensor is None:
        logger_py.warning(
            'Attempt to get debugging tensor before setting debugging mode to true.')
        set_debugging_mode_(True)
    return _debugging_tensor


__all__ = [k for k in globals().keys() if not k.startswith("_")]
