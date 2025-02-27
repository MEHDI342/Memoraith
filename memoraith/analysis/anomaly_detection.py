"""
Anomaly detection functionality for identifying outliers in profiling metrics.
"""

import numpy as np
from typing import Dict, Any, List
import logging

class AnomalyDetector:
    """Detects anomalies in the performance metrics."""

    def __init__(self, z_threshold: float = 3.0):
        """
        Initialize the anomaly detector with configurable threshold.
        """
        self.z_threshold = z_threshold
        self.logger = logging.getLogger(__name__)

    async def detect(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in metrics data using statistical analysis.
        """
        anomalies = []

        try:
            # Extract layer names and metrics
            layers = []
            cpu_memory_values = []
            gpu_memory_values = []
            time_values = []

            for layer, layer_metrics in metrics.items():
                if layer == 'global_metrics' or not isinstance(layer_metrics, dict):
                    continue

                layers.append(layer)
                cpu_memory_values.append(layer_metrics.get('total_cpu_memory', 0))
                gpu_memory_values.append(layer_metrics.get('total_gpu_memory', 0))
                time_values.append(layer_metrics.get('total_time', 0))

            # Detect anomalies in each metric type
            if layers:  # Ensure we have data to analyze
                anomalies.extend(self._detect_anomalies(cpu_memory_values, 'CPU Memory', layers))
                anomalies.extend(self._detect_anomalies(gpu_memory_values, 'GPU Memory', layers))
                anomalies.extend(self._detect_anomalies(time_values, 'Computation Time', layers))

        except Exception as e:
            self.logger.error(f"Error during anomaly detection: {str(e)}")

        return anomalies

    def _detect_anomalies(self, data: List[float], data_type: str, layers: List[str]) -> List[Dict[str, Any]]:
        """Helper method to detect anomalies using z-score."""
        anomalies = []
        if not data or len(data) < 2:  # Need at least 2 data points for meaningful statistics
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
                    'z_score': z_score,
                    'severity': 'high' if abs(z_score) > 2 * self.z_threshold else 'medium'
                })

        return anomalies
