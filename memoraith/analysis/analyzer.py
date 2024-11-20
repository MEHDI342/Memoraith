import logging
from typing import Dict, Any, List
from .metrics import MetricsCalculator
from .bottleneck import BottleneckDetector
from .recommendations import RecommendationEngine
from .anomaly_detection import AnomalyDetector

class AnalyzerManager:
    def __init__(self, profiling_data: Dict[str, Any]):
        self.data = profiling_data
        self.logger = logging.getLogger(__name__)
        self.metrics = MetricsCalculator(profiling_data)
        self.bottleneck_detector = BottleneckDetector()
        self.anomaly_detector = AnomalyDetector()
        self.recommendation_engine = RecommendationEngine()

    async def analyze(self) -> Dict[str, Any]:
        """Run complete analysis on profiling data"""
        try:
            # Calculate basic metrics
            metrics = await self.metrics.calculate()

            # Detect issues and generate recommendations
            bottlenecks = await self.bottleneck_detector.detect(metrics)
            anomalies = await self.anomaly_detector.detect(metrics)
            recommendations = await self.recommendation_engine.generate(metrics)

            # Calculate performance score
            score = self._calculate_performance_score(metrics)

            return {
                'metrics': metrics,
                'bottlenecks': bottlenecks,
                'anomalies': anomalies,
                'recommendations': recommendations,
                'performance_score': score
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        try:
            # Simple scoring based on key metrics
            memory_score = 100 - min(metrics.get('peak_memory_percent', 0), 100)
            cpu_score = 100 - min(metrics.get('peak_cpu_percent', 0), 100)
            network_score = 100 - min(metrics.get('network_utilization_percent', 0), 100)

            # Average the scores
            return (memory_score + cpu_score + network_score) / 3
        except Exception as e:
            self.logger.error(f"Score calculation failed: {str(e)}")
            return 0.0

    async def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the analysis"""
        try:
            analysis = await self.analyze()
            return {
                'performance_score': analysis['performance_score'],
                'bottleneck_count': len(analysis['bottlenecks']),
                'anomaly_count': len(analysis['anomalies']),
                'recommendation_count': len(analysis['recommendations']),
                'peak_memory': analysis['metrics'].get('peak_memory', 0),
                'peak_cpu': analysis['metrics'].get('peak_cpu', 0),
                'total_time': analysis['metrics'].get('total_time', 0)
            }
        except Exception as e:
            self.logger.error(f"Summary generation failed: {str(e)}")
            return {}