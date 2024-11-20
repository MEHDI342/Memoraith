
import logging
from typing import Dict, Any

class MetricsCalculator:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.logger = logging.getLogger(__name__)

    async def calculate(self) -> Dict[str, Any]:
        """Calculate key performance metrics"""
        try:
            # Process raw data
            network_metrics = self._calculate_network_metrics()
            memory_metrics = self._calculate_memory_metrics()
            time_metrics = self._calculate_time_metrics()

            return {
                **network_metrics,
                **memory_metrics,
                **time_metrics,
                'global_metrics': self._calculate_global_metrics()
            }
        except Exception as e:
            self.logger.error(f"Metrics calculation failed: {str(e)}")
            return {}

    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate network-related metrics"""
        network_data = self.data.get('network', {})
        return {
            'peak_bandwidth': max((m.get('bandwidth_mbps', 0) for m in network_data), default=0),
            'average_bandwidth': sum((m.get('bandwidth_mbps', 0) for m in network_data)) / len(network_data) if network_data else 0,
            'total_bytes_transferred': sum((m.get('bytes_sent', 0) + m.get('bytes_recv', 0) for m in network_data))
        }

    def _calculate_memory_metrics(self) -> Dict[str, float]:
        """Calculate memory-related metrics"""
        memory_data = self.data.get('memory', {})
        return {
            'peak_memory': max((m.get('rss', 0) for m in memory_data), default=0),
            'average_memory': sum((m.get('rss', 0) for m in memory_data)) / len(memory_data) if memory_data else 0,
            'peak_memory_percent': max((m.get('memory_percent', 0) for m in memory_data), default=0)
        }

    def _calculate_time_metrics(self) -> Dict[str, float]:
        """Calculate time-related metrics"""
        time_data = self.data.get('time', {})
        return {
            'total_time': sum((m.get('duration', 0) for m in time_data)),
            'average_response_time': sum((m.get('duration', 0) for m in time_data)) / len(time_data) if time_data else 0
        }

    def _calculate_global_metrics(self) -> Dict[str, Any]:
        """Calculate global performance metrics"""
        return {
            'start_time': min((m.get('timestamp', 0) for m in self.data.get('time', [])), default=0),
            'end_time': max((m.get('timestamp', 0) for m in self.data.get('time', [])), default=0),
            'total_samples': len(self.data.get('time', [])),
            'success_rate': self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate the overall success rate of operations"""
        total = len(self.data.get('time', []))
        if not total:
            return 0.0
        failures = sum(1 for m in self.data.get('time', []) if m.get('error', False))
        return (total - failures) / total * 100