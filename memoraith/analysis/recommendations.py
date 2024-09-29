from typing import Dict, Any, List
import logging

class RecommendationEngine:
    """Provides optimization suggestions based on detected bottlenecks."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def generate(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate recommendations for each bottleneck.

        Args:
            metrics (Dict[str, Any]): Calculated metrics for each layer

        Returns:
            List[Dict[str, str]]: List of recommendations
        """
        recommendations = []
        try:
            for layer, layer_metrics in metrics.items():
                if not isinstance(layer_metrics, dict):
                    continue

                layer_recommendations = await self.generate_for_layer(layer, layer_metrics)
                recommendations.extend(layer_recommendations)
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")

        return recommendations

    async def generate_for_layer(self, layer: str, layer_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate recommendations for a specific layer.

        Args:
            layer (str): Name of the layer
            layer_metrics (Dict[str, Any]): Metrics for the specific layer

        Returns:
            List[Dict[str, str]]: List of recommendations for the layer
        """
        recommendations = []

        try:
            if layer_metrics['total_time'] > 0.1:  # Arbitrary threshold
                recommendations.append({
                    'layer': layer,
                    'recommendation': f"Consider optimizing the {layer} for speed. It's taking {layer_metrics['total_time']:.4f} seconds."
                })

            if layer_metrics['total_cpu_memory'] > 1000:  # Arbitrary threshold (1000 MB)
                recommendations.append({
                    'layer': layer,
                    'recommendation': f"The {layer} is using a lot of memory ({layer_metrics['total_cpu_memory']:.2f} MB). Consider reducing its size or using more memory-efficient operations."
                })

            if 'parameters' in layer_metrics and layer_metrics['parameters'] > 1000000:  # Arbitrary threshold (1M parameters)
                recommendations.append({
                    'layer': layer,
                    'recommendation': f"The {layer} has a large number of parameters ({layer_metrics['parameters']:,}). Consider using techniques like pruning or quantization to reduce model size."
                })
        except KeyError as e:
            self.logger.error(f"Missing key in layer metrics for {layer}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating recommendations for {layer}: {str(e)}")

        return recommendations