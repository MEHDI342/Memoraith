# memoraith/analysis/analyzer.py
from typing import Dict, Any, List
import logging
import numpy as np
from .metrics import MetricsCalculator
from .bottleneck import BottleneckDetector
from .recommendations import RecommendationEngine
from .anomaly_detection import AnomalyDetector

class Analyzer:
    """Main analysis coordinator for profiling data."""

    def __init__(self, profiling_data: Dict[str, Any]):
        self.data = profiling_data
        self.logger = logging.getLogger(__name__)
        self.metrics_calculator = MetricsCalculator(profiling_data)
        self.bottleneck_detector = BottleneckDetector()
        self.anomaly_detector = AnomalyDetector()
        self.recommendation_engine = RecommendationEngine()

    async def run_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        try:
            metrics = await self.metrics_calculator.calculate()
            if not metrics:
                raise ValueError("No metrics calculated")

            bottlenecks = await self.bottleneck_detector.detect(metrics)
            anomalies = await self.anomaly_detector.detect(metrics)
            recommendations = await self.recommendation_engine.generate(metrics)
            performance_score = self._calculate_performance_score(metrics)

            return {
                'metrics': metrics,
                'bottlenecks': bottlenecks,
                'anomalies': anomalies,
                'recommendations': recommendations,
                'performance_score': performance_score,
                'summary': await self._generate_summary(metrics)
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score."""
        try:
            # Weight factors for different metrics
            weights = {
                'memory': 0.4,
                'compute': 0.4,
                'efficiency': 0.2
            }

            # Memory score (lower is better)
            memory_usage = metrics.get('peak_memory_percent', 0)
            memory_score = max(0, 100 - memory_usage)

            # Compute score (based on time efficiency)
            compute_time = metrics.get('total_time', 0)
            compute_score = 100 * np.exp(-compute_time / 10)  # Exponential decay

            # Efficiency score (based on resource utilization)
            efficiency_score = metrics.get('resource_efficiency', 80)

            # Calculate weighted score
            final_score = (
                    weights['memory'] * memory_score +
                    weights['compute'] * compute_score +
                    weights['efficiency'] * efficiency_score
            )

            return min(100, max(0, final_score))
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {str(e)}")
            return 0.0

    async def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a concise summary of analysis results."""
        return {
            'total_time': metrics.get('total_time', 0),
            'peak_memory': metrics.get('peak_memory', 0),
            'gpu_utilization': metrics.get('gpu_utilization', 0),
            'bottleneck_count': len(metrics.get('bottlenecks', [])),
            'anomaly_count': len(metrics.get('anomalies', [])),
            'recommendation_count': len(metrics.get('recommendations', []))
        }

    async def export_results(self, format: str = 'json') -> Dict[str, Any]:
        """Export analysis results in specified format."""
        try:
            results = await self.run_analysis()
            if format == 'json':
                return results
            elif format == 'summary':
                return await self._generate_summary(results['metrics'])
            else:
                raise ValueError(f"Unsupported export format: {format}")
        except Exception as e:
            self.logger.error(f"Error exporting results: {str(e)}")
            raise