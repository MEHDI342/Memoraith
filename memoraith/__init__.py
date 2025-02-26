"""
Memoraith: Advanced Lightweight Model Profiler for Deep Learning
----------------------------------------------------------------

Memoraith provides unparalleled insights into neural network performance through
detailed profiling of memory usage, computation time, and resource utilization.
"""

__version__ = '0.5.0'

from memoraith.profiler import profile_model, set_output_path
from memoraith.config import Config
from memoraith.exceptions import MemoraithError

__all__ = [
    'profile_model',
    'set_output_path',
    'Config',
    'MemoraithError',
    '__version__',
]