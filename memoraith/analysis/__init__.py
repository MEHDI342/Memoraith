"""
Memoraith Analysis Module - Core functionality for analyzing profiling data.
"""

from .metrics import MetricsCalculator
from .bottleneck import BottleneckDetector
from .recommendations import RecommendationEngine
from .anomaly_detection import AnomalyDetector

__all__ = [
    'Analyzer',
    'MetricsCalculator',
    'BottleneckDetector',
    'RecommendationEngine',
    'AnomalyDetector'
]

class Analyzer:
    """Main analysis coordinator for profiling data."""

    def __init__(self, profiling_data):
        """
        Initialize the Analyzer with profiling data.
        """
        from .analyzer import Analyzer as AnalyzerImpl
        self._impl = AnalyzerImpl(profiling_data)

    async def run_analysis(self):
        """
        Run complete analysis pipeline.
        """
        return await self._impl.run_analysis()

    async def export_results(self, format='json'):
        """
        Export analysis results in specified format.
        """
        return await self._impl.export_results(format)
