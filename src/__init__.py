import os

_SRC_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_MODELS = os.path.join(_PROJECT_ROOT, "checkpoints")  # root of models


def get_device():
    if torch.cuda.is_available():
        logging.info("Using CUDA")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        logging.info("Using MPS")
        return torch.device('mps')
    else:
        logging.info("Using CPU")
        return torch.device('cpu')