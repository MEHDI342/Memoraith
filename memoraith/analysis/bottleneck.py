from typing import Dict, Any, List
import logging

class BottleneckDetector:
    """Identifies layers or operations that are bottlenecks."""

    def __init__(self, time_threshold: float = 0.1, memory_threshold: float = 0.2):
        self.time_threshold = time_threshold
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)

    async def detect(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect bottlenecks in the model based on time and memory usage.

        Args:
            metrics (Dict[str, Any]): Calculated metrics for each layer

        Returns:
            List[Dict[str, Any]]: List of detected bottlenecks
        """
        bottlenecks = []
        total_time = sum(layer['total_time'] for layer in metrics.values() if isinstance(layer, dict))
        total_memory = sum(layer['total_cpu_memory'] for layer in metrics.values() if isinstance(layer, dict))

        try:
            for layer, layer_metrics in metrics.items():
                if not isinstance(layer_metrics, dict):
                    continue

                time_ratio = layer_metrics['total_time'] / total_time if total_time else 0
                memory_ratio = layer_metrics['total_cpu_memory'] / total_memory if total_memory else 0

                if time_ratio > self.time_threshold:
                    bottlenecks.append({
                        'layer': layer,
                        'type': 'time',
                        'value': layer_metrics['total_time'],
                        'ratio': time_ratio
                    })

                if memory_ratio > self.memory_threshold:
                    bottlenecks.append({
                        'layer': layer,
                        'type': 'memory',
                        'value': layer_metrics['total_cpu_memory'],
                        'ratio': memory_ratio
                    })
        except Exception as e:
            self.logger.error(f"Error during bottleneck detection: {str(e)}")

        return bottlenecks

    async def detect_for_layer(self, layer_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect bottlenecks for a specific layer.

        Args:
            layer_metrics (Dict[str, Any]): Metrics for a specific layer

        Returns:
            List[Dict[str, Any]]: List of detected bottlenecks for the layer
        """
        bottlenecks = []

        try:
            if layer_metrics['total_time'] > self.time_threshold:
                bottlenecks.append({
                    'type': 'time',
                    'value': layer_metrics['total_time']
                })

            if layer_metrics['total_cpu_memory'] > self.memory_threshold:
                bottlenecks.append({
                    'type': 'memory',
                    'value': layer_metrics['total_cpu_memory']
                })
        except KeyError as e:
            self.logger.error(f"Missing key in layer metrics: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error during layer bottleneck detection: {str(e)}")

        return bottlenecks