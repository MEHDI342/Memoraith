"""
Memoraith: Lightweight Model Profiler
"""

__version__ = '0.5.0'

from memoraith.profiler import profile_model, set_output_path
from memoraith.config import Config
from memoraith.exceptions import MemoraithError
from memoraith.analysis import analyzer
# Expose main components
__all__ = [
    'profile_model',
    'set_output_path',
    'Config',
    'MemoraithError',
]