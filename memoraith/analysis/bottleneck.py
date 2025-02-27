"""
Bottleneck detection functionality for identifying performance bottlenecks in a model.
"""

import logging
import numpy as np
from typing import Dict, Any, List

class BottleneckDetector:
    """Detects performance bottlenecks in profiling metrics."""

    def __init__(self, time_threshold: float = 0.1, memory_threshold: float = 0.2):
        """
        Initialize the bottleneck detector with configurable thresholds.
        """
        self.time_threshold = time_threshold
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)

    async def detect(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks in metrics data.
        """
        bottlenecks = []
        try:
            # Extract time and memory metrics
            time_metrics = {}
            memory_metrics = {}

            # Process layer metrics
            for component, values in metrics.items():
                if component == 'global_metrics':
                    continue

                if isinstance(values, dict):
                    if 'total_time' in values:
                        time_metrics[component] = values['total_time']
                    if 'total_cpu_memory' in values:
                        memory_metrics[component] = values['total_cpu_memory']
                    if 'total_gpu_memory' in values:
                        # Use GPU memory if it's higher than CPU memory
                        if values.get('total_gpu_memory', 0) > memory_metrics.get(component, 0):
                            memory_metrics[component] = values['total_gpu_memory']

            # Calculate totals
            total_time = sum(time_metrics.values())
            total_memory = sum(memory_metrics.values())

            # Detect time bottlenecks
            for component, time_value in time_metrics.items():
                if total_time > 0:
                    time_ratio = time_value / total_time
                    if time_ratio > self.time_threshold:
                        bottlenecks.append({
                            'layer': component,
                            'type': 'time',
                            'value': time_value,
                            'ratio': time_ratio,
                            'severity': 'high' if time_ratio > 0.5 else 'medium'
                        })

            # Detect memory bottlenecks
            for component, memory_value in memory_metrics.items():
                if total_memory > 0:
                    memory_ratio = memory_value / total_memory
                    if memory_ratio > self.memory_threshold:
                        bottlenecks.append({
                            'layer': component,
                            'type': 'memory',
                            'value': memory_value,
                            'ratio': memory_ratio,
                            'severity': 'high' if memory_ratio > 0.5 else 'medium'
                        })

            return bottlenecks
        except Exception as e:
            self.logger.error(f"Bottleneck detection failed: {str(e)}")
            return []
