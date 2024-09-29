from typing import Any, Dict, List
import inspect
import logging

logger = logging.getLogger(__name__)

def identify_framework(model: Any) -> str:
    """
    Identify the deep learning framework of the given model.

    Args:
        model: The machine learning model to identify

    Returns:
        str: The identified framework name ('pytorch', 'tensorflow', 'unknown')
    """
    try:
        model_type = type(model).__module__.split('.')[0]
        if model_type == 'torch':
            return 'pytorch'
        elif model_type in ['tensorflow', 'keras']:
            return 'tensorflow'
        else:
            logger.warning(f"Unknown framework for model type: {model_type}")
            return 'unknown'
    except Exception as e:
        logger.error(f"Error identifying framework: {str(e)}")
        return 'unknown'

def get_model_structure(model: Any) -> Dict[str, Any]:
    """
    Get a structured representation of the model's architecture.

    Args:
        model: The machine learning model

    Returns:
        Dict[str, Any]: A dictionary representing the model's structure
    """
    try:
        framework = identify_framework(model)
        if framework == 'pytorch':
            return _get_pytorch_structure(model)
        elif framework == 'tensorflow':
            return _get_tensorflow_structure(model)
        else:
            return {'error': 'Unsupported framework'}
    except Exception as e:
        logger.error(f"Error getting model structure: {str(e)}")
        return {'error': str(e)}

def _get_pytorch_structure(model: Any) -> Dict[str, Any]:
    """Helper function to get PyTorch model structure."""
    structure = {}
    for name, module in model.named_modules():
        if list(module.children()):  # Skip container modules
            continue
        structure[name] = {
            'type': type(module).__name__,
            'parameters': sum(p.numel() for p in module.parameters()),
            'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
        }
    return structure

def _get_tensorflow_structure(model: Any) -> Dict[str, Any]:
    """Helper function to get TensorFlow model structure."""
    structure = {}
    for layer in model.layers:
        structure[layer.name] = {
            'type': type(layer).__name__,
            'parameters': layer.count_params(),
            'trainable': sum(tf.keras.backend.count_params(w) for w in layer.trainable_weights)
        }
    return structure

def estimate_model_size(model: Any) -> Dict[str, float]:
    """
    Estimate the size of the model in memory.

    Args:
        model: The machine learning model

    Returns:
        Dict[str, float]: Estimated size in MB for parameters and buffers
    """
    try:
        framework = identify_framework(model)
        if framework == 'pytorch':
            return _estimate_pytorch_size(model)
        elif framework == 'tensorflow':
            return _estimate_tensorflow_size(model)
        else:
            return {'error': 'Unsupported framework'}
    except Exception as e:
        logger.error(f"Error estimating model size: {str(e)}")
        return {'error': str(e)}

def _estimate_pytorch_size(model: Any) -> Dict[str, float]:
    """Helper function to estimate PyTorch model size."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        'parameters': param_size / (1024 * 1024),
        'buffers': buffer_size / (1024 * 1024)
    }

def _estimate_tensorflow_size(model: Any) -> Dict[str, float]:
    """Helper function to estimate TensorFlow model size."""
    param_size = sum(tf.keras.backend.count_params(w) * w.dtype.size for w in model.weights)
    return {
        'parameters': param_size / (1024 * 1024),
        'buffers': 0  # TensorFlow doesn't have a direct equivalent to PyTorch's buffers
    }

def get_function_info(func: callable) -> Dict[str, Any]:
    """
    Get detailed information about a function.

    Args:
        func: The function to inspect

    Returns:
        Dict[str, Any]: Information about the function
    """
    try:
        signature = inspect.signature(func)
        return {
            'name': func.__name__,
            'module': func.__module__,
            'docstring': inspect.getdoc(func),
            'parameters': [{'name': name, 'annotation': param.annotation.__name__ if param.annotation != inspect.Parameter.empty else None}
                           for name, param in signature.parameters.items()],
            'return_annotation': signature.return_annotation.__name__ if signature.return_annotation != inspect.Signature.empty else None,
            'is_coroutine': inspect.iscoroutinefunction(func),
            'is_generator': inspect.isgeneratorfunction(func),
            'source': inspect.getsource(func)
        }
    except Exception as e:
        logger.error(f"Error getting function info: {str(e)}")
        return {'error': str(e)}