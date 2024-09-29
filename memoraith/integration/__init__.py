from .framework_adapter import FrameworkAdapter
from .pytorch_adapter import PyTorchAdapter
from .tensorflow_adapter import TensorFlowAdapter
from .common_utils import identify_framework
from memoraith.exceptions import FrameworkNotSupportedError

def get_framework_adapter(model):
    """
    Returns the appropriate framework adapter based on the model type.
    
    Args:
        model: The machine learning model to be profiled
    
    Returns:
        FrameworkAdapter: An instance of the appropriate adapter
    
    Raises:
        FrameworkNotSupportedError: If the framework is not supported
    """
    framework_name = identify_framework(model)
    if framework_name == 'pytorch':
        return PyTorchAdapter(model)
    elif framework_name == 'tensorflow':
        return TensorFlowAdapter(model)
    else:
        raise FrameworkNotSupportedError(framework_name)
