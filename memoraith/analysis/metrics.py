from typing import Dict, Any
import logging
import asyncio

class MetricsCalculator:
    """Calculates various performance metrics from the collected data."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.logger = logging.getLogger(__name__)

    async def calculate(self) -> Dict[str, Any]:
        """
        Compute metrics such as total memory and time per layer.

        Returns:
            Dict[str, Any]: Calculated metrics for each layer and global metrics
        """
        metrics = {}
        try:
            for layer_name, stats in self.data.items():
                metrics[layer_name] = await self._calculate_layer_metrics(stats)

            metrics['global'] = await self._calculate_global_metrics(metrics)
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")

        return metrics

    async def _calculate_layer_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for a single layer."""
        return {
            'total_cpu_memory': stats.get('cpu_memory', 0),
            'total_gpu_memory': stats.get('gpu_memory', 0),
            'total_time': stats.get('time', 0),
            'parameters': stats.get('parameters', 0),
            'output_shape': stats.get('output_shape', []),
            'flops': await self._estimate_flops(stats)
        }

    async def _calculate_global_metrics(self, layer_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate global metrics across all layers."""
        total_time = sum(layer['total_time'] for layer in layer_metrics.values() if isinstance(layer, dict))
        total_parameters = sum(layer['parameters'] for layer in layer_metrics.values() if isinstance(layer, dict))
        total_flops = sum(layer['flops'] for layer in layer_metrics.values() if isinstance(layer, dict))

        return {
            'total_time': total_time,
            'total_parameters': total_parameters,
            'peak_cpu_memory': max((layer['total_cpu_memory'] for layer in layer_metrics.values() if isinstance(layer, dict)), default=0),
            'peak_gpu_memory': max((layer['total_gpu_memory'] for layer in layer_metrics.values() if isinstance(layer, dict)), default=0),
            'total_flops': total_flops
        }

    async def _estimate_flops(self, stats: Dict[str, Any]) -> int:
        """Estimate FLOPs for a layer based on its type and parameters."""
        layer_type = stats.get('type', '')
        input_shape = stats.get('input_shape', [])
        output_shape = stats.get('output_shape', [])
        parameters = stats.get('parameters', 0)

        # This is a simplified estimation and should be expanded for different layer types
        if 'conv' in layer_type.lower():
            return parameters * output_shape[1] * output_shape[2]  # Assuming NCHW format
        elif 'linear' in layer_type.lower():
            return 2 * parameters  # multiply-add operations
        else:
            return parameters  # fallback estimation

    async def get_layer_efficiency(self, layer_name: str) -> float:
        """Calculate the efficiency of a layer (FLOPs per second)."""
        layer_metrics = self.data.get(layer_name, {})
        flops = await self._estimate_flops(layer_metrics)
        time = layer_metrics.get('time', 1e-6)  # Avoid division by zero
        return flops / time if time > 0 else 0

    async def get_model_efficiency(self) -> float:
        """Calculate the overall model efficiency (FLOPs per second)."""
        metrics = await self.calculate()
        total_flops = metrics['global']['total_flops']
        total_time = metrics['global']['total_time']
        return total_flops / total_time if total_time > 0 else 0