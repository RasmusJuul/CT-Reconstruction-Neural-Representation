import os
from torch import device
from torch.cuda import is_available as is_cuda_available


_SRC_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_MODELS = os.path.join(_PROJECT_ROOT, "checkpoints_")  # root of models


def get_device():
    if is_cuda_available():
        return device("cuda")
    else:
        return device("cpu")
