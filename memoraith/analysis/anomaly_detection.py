import numpy as np
from typing import Dict, Any, List
import logging

class AnomalyDetector:
    """Detects anomalies in the performance metrics."""

    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold
        self.logger = logging.getLogger(__name__)

    async def detect(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies such as sudden spikes in memory or time.

        Args:
            metrics (Dict[str, Any]): Calculated metrics for each layer

        Returns:
            List[Dict[str, Any]]: List of detected anomalies
        """
        anomalies = []
        layers = list(metrics.keys())

        try:
            cpu_memory = [metrics[layer].get('total_cpu_memory', 0) for layer in layers]
            gpu_memory = [metrics[layer].get('total_gpu_memory', 0) for layer in layers]
            time = [metrics[layer].get('total_time', 0) for layer in layers]

            anomalies.extend(self._detect_anomalies(cpu_memory, 'CPU Memory', layers))
            anomalies.extend(self._detect_anomalies(gpu_memory, 'GPU Memory', layers))
            anomalies.extend(self._detect_anomalies(time, 'Computation Time', layers))
        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {str(e)}")

        return anomalies

    def _detect_anomalies(self, data: List[float], data_type: str, layers: List[str]) -> List[Dict[str, Any]]:
        """Helper method to detect anomalies using z-score."""
        anomalies = []
        if not data:
            return anomalies

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return anomalies  # No variation, so no anomalies

        z_scores = [(x - mean) / std for x in data]

        for layer, z_score, value in zip(layers, z_scores, data):
            if abs(z_score) > self.z_threshold:
                anomalies.append({
                    'layer': layer,
                    'type': data_type,
                    'value': value,
                    'z_score': z_score
                })

        return anomalies