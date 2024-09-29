from typing import Dict, Any, List, Optional
import asyncio
import logging
from .metrics import MetricsCalculator
from .bottleneck import BottleneckDetector
from .recommendations import RecommendationEngine
from .anomaly_detection import AnomalyDetector

class Analyzer:
    """Complete analyzer for profiling data to identify bottlenecks and suggest optimizations."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.metrics = MetricsCalculator(data)
        self.bottleneck_detector = BottleneckDetector()
        self.recommendation_engine = RecommendationEngine()
        self.anomaly_detector = AnomalyDetector()
        self.logger = logging.getLogger(__name__)

    async def run_analysis(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline asynchronously."""
        try:
            metrics = await self.metrics.calculate()

            bottlenecks, anomalies, recommendations = await asyncio.gather(
                self.bottleneck_detector.detect(metrics),
                self.anomaly_detector.detect(metrics),
                self.recommendation_engine.generate(metrics)
            )

            performance_score = await self.get_overall_performance_score(metrics)

            return {
                'metrics': metrics,
                'bottlenecks': bottlenecks,
                'anomalies': anomalies,
                'recommendations': recommendations,
                'performance_score': performance_score
            }
        except Exception as e:
            self.logger.error(f"Error during analysis: {str(e)}")
            raise

    async def get_layer_analysis(self, layer_name: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific layer."""
        metrics = await self.metrics.calculate()
        layer_metrics = metrics.get(layer_name, {})

        if not layer_metrics:
            return {'error': f"Layer '{layer_name}' not found in metrics."}

        bottlenecks = await self.bottleneck_detector.detect_for_layer(layer_metrics)
        anomalies = await self.anomaly_detector.detect_for_layer(layer_metrics)
        recommendations = await self.recommendation_engine.generate_for_layer(layer_metrics)

        return {
            'metrics': layer_metrics,
            'bottlenecks': bottlenecks,
            'anomalies': anomalies,
            'recommendations': recommendations
        }

    async def get_top_bottlenecks(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the top N bottlenecks across all layers."""
        metrics = await self.metrics.calculate()
        all_bottlenecks = await self.bottleneck_detector.detect(metrics)
        sorted_bottlenecks = sorted(all_bottlenecks, key=lambda x: x.get('severity', 0), reverse=True)
        return sorted_bottlenecks[:n]

    async def get_overall_performance_score(self, metrics: Optional[Dict[str, Any]] = None) -> float:
        """Calculate an overall performance score based on various metrics."""
        if metrics is None:
            metrics = await self.metrics.calculate()

        total_time = sum(layer['total_time'] for layer in metrics.values() if isinstance(layer, dict))
        total_memory = sum(layer['total_cpu_memory'] for layer in metrics.values() if isinstance(layer, dict))
        num_bottlenecks = len(await self.bottleneck_detector.detect(metrics))
        num_anomalies = len(await self.anomaly_detector.detect(metrics))

        # Lower times, memory usage, and fewer bottlenecks/anomalies result in a higher score
        score = 100 - (total_time * 10 + total_memory / 100 + num_bottlenecks * 5 + num_anomalies * 3)
        return max(0, min(score, 100))  # Ensure the score is between 0 and 100

    async def compare_runs(self, other_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare the current run with another run."""
        current_metrics = await self.metrics.calculate()
        other_analyzer = Analyzer(other_data)
        other_metrics = await other_analyzer.metrics.calculate()

        comparison = {}
        for layer in set(current_metrics.keys()) | set(other_metrics.keys()):
            comparison[layer] = {
                'current': current_metrics.get(layer, {}),
                'other': other_metrics.get(layer, {}),
                'time_diff': current_metrics.get(layer, {}).get('total_time', 0) - other_metrics.get(layer, {}).get('total_time', 0),
                'memory_diff': current_metrics.get(layer, {}).get('total_cpu_memory', 0) - other_metrics.get(layer, {}).get('total_cpu_memory', 0)
            }
        return comparison

    async def get_memory_profile(self) -> Dict[str, Any]:
        """Get a detailed memory profile of the model."""
        metrics = await self.metrics.calculate()
        total_memory = sum(layer['total_cpu_memory'] for layer in metrics.values() if isinstance(layer, dict))
        peak_memory = max(layer['total_cpu_memory'] for layer in metrics.values() if isinstance(layer, dict))

        memory_intensive_layers = sorted(
            [(layer, metrics[layer]['total_cpu_memory']) for layer in metrics if isinstance(metrics[layer], dict)],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_memory': total_memory,
            'peak_memory': peak_memory,
            'memory_intensive_layers': memory_intensive_layers
        }

    async def get_time_profile(self) -> Dict[str, Any]:
        """Get a detailed time profile of the model."""
        metrics = await self.metrics.calculate()
        total_time = sum(layer['total_time'] for layer in metrics.values() if isinstance(layer, dict))

        time_intensive_layers = sorted(
            [(layer, metrics[layer]['total_time']) for layer in metrics if isinstance(metrics[layer], dict)],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_time': total_time,
            'time_intensive_layers': time_intensive_layers
        }

    async def generate_optimization_plan(self) -> List[Dict[str, Any]]:
        """Generate a step-by-step optimization plan based on the analysis."""
        analysis_results = await self.run_analysis()
        optimization_plan = []
        for bottleneck in analysis_results['bottlenecks']:
            recommendation = next((r for r in analysis_results['recommendations'] if r['layer'] == bottleneck['layer']), None)
            if recommendation:
                optimization_plan.append({
                    'step': len(optimization_plan) + 1,
                    'layer': bottleneck['layer'],
                    'issue': bottleneck['type'],
                    'recommendation': recommendation['recommendation'],
                    'expected_improvement': 'Medium to High'  # This could be more sophisticated
                })
        return optimization_plan

    async def get_layer_efficiency(self) -> Dict[str, float]:
        """Calculate and return the efficiency of each layer."""
        metrics = await self.metrics.calculate()
        efficiency = {}
        for layer, layer_metrics in metrics.items():
            if isinstance(layer_metrics, dict):
                time = layer_metrics.get('total_time', 1e-6)  # Avoid division by zero
                params = layer_metrics.get('parameters', 0)
                efficiency[layer] = params / time if time > 0 else 0
        return efficiency

    async def get_model_complexity(self) -> Dict[str, Any]:
        """Analyze and return the complexity of the model."""
        metrics = await self.metrics.calculate()
        total_params = sum(layer['parameters'] for layer in metrics.values() if isinstance(layer, dict))
        total_flops = sum(layer.get('flops', 0) for layer in metrics.values() if isinstance(layer, dict))

        return {
            'total_parameters': total_params,
            'total_flops': total_flops,
            'complexity_score': total_flops / total_params if total_params > 0 else 0
        }

    async def get_training_efficiency(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the efficiency of the training process."""
        total_epochs = len(training_data)
        total_time = sum(epoch['epoch_time'] for epoch in training_data)
        avg_time_per_epoch = total_time / total_epochs if total_epochs > 0 else 0

        return {
            'total_epochs': total_epochs,
            'total_training_time': total_time,
            'avg_time_per_epoch': avg_time_per_epoch,
            'training_speed': 1 / avg_time_per_epoch if avg_time_per_epoch > 0 else 0
        }

    def __str__(self) -> str:
        """String representation of the Analyzer."""
        return f"Analyzer(data_size={len(self.data)})"

    def __repr__(self) -> str:
        """Representation of the Analyzer."""
        return self.__str__()