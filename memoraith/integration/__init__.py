from .framework_adapter import FrameworkAdapter
from .pytorch_adapter import PyTorchAdapter
from .tensorflow_adapter import TensorFlowAdapter
from .common_utils import identify_framework

def get_framework_adapter(model):
    framework_name = identify_framework(model)
    if framework_name == 'pytorch':
        return PyTorchAdapter(model)
    elif framework_name == 'tensorflow':
        return TensorFlowAdapter(model)
    else:
        from memoraith.exceptions import FrameworkNotSupportedError
        raise FrameworkNotSupportedError(framework_name)