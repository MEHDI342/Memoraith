"""
Memoraith: Advanced Lightweight Model Profiler for Deep Learning
----------------------------------------------------------------

Memoraith provides unparalleled insights into neural network performance through
detailed profiling of memory usage, computation time, and resource utilization.
"""

__version__ = '0.5.0'

# Import core exceptions first to avoid circular imports
from memoraith.exceptions import (
    MemoraithError,
    FrameworkNotSupportedError,
    ConfigurationError,
    ProfilingError,
    DataCollectionError,
    AnalysisError,
    ReportGenerationError,
    GPUNotAvailableError
)

# Import configuration module
from memoraith.config import Config, config

# Import main profiling functions
from memoraith.profiler import profile_model, set_output_path

__all__ = [
    'profile_model',
    'set_output_path',
    'Config',
    'config',
    'MemoraithError',
    'FrameworkNotSupportedError',
    'ConfigurationError',
    'ProfilingError',
    'DataCollectionError',
    'AnalysisError',
    'ReportGenerationError',
    'GPUNotAvailableError',
    '__version__',
]