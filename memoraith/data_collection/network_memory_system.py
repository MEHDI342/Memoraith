"""
Complete Network and Memory Profiling System
Filename: memoraith/data_collection/network_memory_system.py
"""

import os
import psutil
import threading
import tarfile
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
from enum import Enum, auto
from contextlib import asynccontextmanager, contextmanager
import json
from pathlib import Path
import aiofiles
import numpy as np
from datetime import datetime

# Core Exceptions
class ProfilingError(Exception):
    """to do base exception for profiling errors"""
    pass

class ResourceError(ProfilingError):
    """to do Resource management related errors"""
    pass

class NetworkError(ProfilingError):
    """to do Network-specific profiling errors"""
    pass

class MemoryError(ProfilingError):
    """to do Memory-specific profiling errors"""
    pass

# Configuration Classes
class ProfilingType(Enum):
    NETWORK = auto()
    MEMORY = auto()
    BOTH = auto()

@dataclass
class ProfilingConfig:
    """Configuration for the profiling system"""
    interval: float = 0.1
    detailed: bool = True
    log_directory: str = "profiling_logs"
    max_samples: int = 10000
    enable_disk_logging: bool = True
    enable_console_output: bool = True
    async_mode: bool = True
    memory_threshold_mb: float = 1000.0
    network_threshold_mbps: float = 100.0
    alert_enabled: bool = True
    compression_enabled: bool = True
    backup_enabled: bool = True
    retention_days: int = 7

class MetricCollector:
    """Base class for metric collection"""

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self.is_collecting = False
        self.start_time: Optional[float] = None

    async def start(self) -> None:
        """Start metric collection"""
        raise NotImplementedError

    async def stop(self) -> None:
        """Stop metric collection"""
        raise NotImplementedError

    async def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics"""
        raise NotImplementedError

    def _setup_logging(self, name: str) -> None:
        """Setup logging configuration"""
        log_path = Path(self.config.log_directory) / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.config.enable_disk_logging:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if self.config.enable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.DEBUG)

class NetworkMetrics(MetricCollector):
    """Network metrics collection implementation"""

    def __init__(self, config: ProfilingConfig):
        super().__init__(config)
        self._setup_logging("network")
        self.baseline_counters: Optional[psutil._common.snetio] = None
        self.task: Optional[asyncio.Task] = None
        self._setup_storage()

    def _setup_storage(self) -> None:
        """Initialize storage for network metrics"""
        self.storage_path = Path(self.config.log_directory) / "network_metrics"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start network metrics collection"""
        try:
            async with self._async_lock:
                if self.is_collecting:
                    raise NetworkError("Network metrics collection already running")

                self.start_time = time.time()
                self.is_collecting = True
                self.metrics.clear()
                self.baseline_counters = psutil.net_io_counters()

                if self.config.async_mode:
                    self.task = asyncio.create_task(self._collect_metrics_async())
                else:
                    threading.Thread(
                        target=self._collect_metrics_sync,
                        daemon=True
                    ).start()

                self.logger.info("Network metrics collection started")
                await self._save_state("started")

        except Exception as e:
            self.logger.error(f"Failed to start network metrics collection: {e}", exc_info=True)
            self.is_collecting = False
            raise NetworkError(f"Failed to start network collection: {str(e)}") from e

    async def _save_state(self, state: str) -> None:
        """
        Save profiler state to disk for monitoring and recovery.

        Args:
            state: Current operational state
        """
        try:
            state_file = self.storage_path / "profiler_state.json"
            state_data = {
                "timestamp": time.time(),
                "state": state,
                "collection_start_time": self.start_time,
                "samples_collected": len(self.metrics),
                "config": {k: str(v) if isinstance(v, Path) else v
                           for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }

            # Atomic write using temporary file
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            # Atomic rename
            temp_file.replace(state_file)
            self.logger.debug(f"Network metrics state saved: {state}")

        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to save state: {str(e)}")

    async def stop(self) -> Dict[str, Any]:
        """Stop network metrics collection and return results"""
        try:
            async with self._async_lock:
                if not self.is_collecting:
                    raise NetworkError("Network metrics collection not running")

                self.is_collecting = False
                if self.task:
                    await self.task
                    self.task = None

                results = await self._generate_report()
                await self._save_state("stopped")
                return results

        except Exception as e:
            self.logger.error(f"Error stopping network metrics collection: {e}", exc_info=True)
            raise NetworkError(f"Failed to stop network collection: {str(e)}") from e

    async def _collect_metrics_async(self) -> None:
        """Asynchronous network metrics collection"""
        sample_count = 0
        last_counters = self.baseline_counters

        while self.is_collecting and sample_count < self.config.max_samples:
            try:
                await asyncio.sleep(self.config.interval)
                current_counters = psutil.net_io_counters()

                metrics = await self._calculate_metrics(last_counters, current_counters)
                await self._store_metrics(metrics)

                if await self._should_alert(metrics):
                    await self._send_alert(metrics)

                last_counters = current_counters
                sample_count += 1

            except Exception as e:
                self.logger.error(f"Error in async network collection: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _collect_metrics_sync(self) -> None:
        """Synchronous network metrics collection"""
        sample_count = 0
        last_counters = self.baseline_counters

        while self.is_collecting and sample_count < self.config.max_samples:
            try:
                time.sleep(self.config.interval)
                current_counters = psutil.net_io_counters()

                metrics = self._calculate_metrics_sync(last_counters, current_counters)
                self._store_metrics_sync(metrics)

                if self._should_alert_sync(metrics):
                    self._send_alert_sync(metrics)

                last_counters = current_counters
                sample_count += 1

            except Exception as e:
                self.logger.error(f"Error in sync network collection: {e}", exc_info=True)
                time.sleep(1)

    def _should_alert_sync(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics should trigger an alert in synchronous mode.

        Args:
            metrics: Current metrics data point

        Returns:
            bool: True if alert threshold exceeded
        """
        if not self.config.alert_enabled:
            return False

        # Check primary bandwidth threshold
        if metrics['bandwidth_mbps'] > self.config.network_threshold_mbps:
            return True

        # Check error rate if detailed metrics enabled
        if self.config.detailed_metrics and metrics.get('error_rate', 0) > 5.0:  # 5% error rate
            return True

        return False

    def _send_alert_sync(self, metrics: Dict[str, Any]) -> None:
        """
        Send alert for concerning metrics in synchronous mode.

        Args:
            metrics: Metrics that triggered the alert
        """
        alert_message = (
            f"Network Alert: Bandwidth {metrics['bandwidth_mbps']:.2f} Mbps "
            f"exceeds threshold {self.config.network_threshold_mbps} Mbps"
        )
        self.logger.warning(alert_message)
        # Additional alert mechanisms could be added here (email, SMS, etc.)

class MemoryMetrics(MetricCollector):
    """Memory metrics collection implementation"""

    def __init__(self, config: ProfilingConfig):
        super().__init__(config)
        self._setup_logging("memory")
        self.task: Optional[asyncio.Task] = None
        self._setup_storage()
        self.process = psutil.Process()

    def _setup_storage(self) -> None:
        """Initialize storage for memory metrics"""
        self.storage_path = Path(self.config.log_directory) / "memory_metrics"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start memory metrics collection"""
        try:
            async with self._async_lock:
                if self.is_collecting:
                    raise MemoryError("Memory metrics collection already running")

                self.start_time = time.time()
                self.is_collecting = True
                self.metrics.clear()

                if self.config.async_mode:
                    self.task = asyncio.create_task(self._collect_metrics_async())
                else:
                    threading.Thread(
                        target=self._collect_metrics_sync,
                        daemon=True
                    ).start()

                self.logger.info("Memory metrics collection started")
                await self._save_state("started")

        except Exception as e:
            self.logger.error(f"Failed to start memory metrics collection: {e}", exc_info=True)
            self.is_collecting = False
            raise MemoryError(f"Failed to start memory collection: {str(e)}") from e

    async def _save_state(self, state: str) -> None:
        """
        Save profiler state to disk for monitoring and recovery.

        Args:
            state: Current operational state
        """
        try:
            state_file = self.storage_path / "profiler_state.json"
            state_data = {
                "timestamp": time.time(),
                "state": state,
                "collection_start_time": self.start_time,
                "samples_collected": len(self.metrics),
                "config": {k: str(v) if isinstance(v, Path) else v
                           for k, v in self.config.__dict__.items() if not k.startswith('_')}
            }

            # Atomic write using temporary file
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            # Atomic rename
            temp_file.replace(state_file)
            self.logger.debug(f"Memory metrics state saved: {state}")

        except (OSError, IOError) as e:
            self.logger.warning(f"Failed to save state: {str(e)}")

    async def stop(self) -> Dict[str, Any]:
        """Stop memory metrics collection and return results"""
        try:
            async with self._async_lock:
                if not self.is_collecting:
                    raise MemoryError("Memory metrics collection not running")

                self.is_collecting = False
                if self.task:
                    await self.task
                    self.task = None

                results = await self._generate_report()
                await self._save_state("stopped")
                return results

        except Exception as e:
            self.logger.error(f"Error stopping memory metrics collection: {e}", exc_info=True)
            raise MemoryError(f"Failed to stop memory collection: {str(e)}") from e

    async def _collect_metrics_async(self) -> None:
        """Asynchronous memory metrics collection"""
        sample_count = 0

        while self.is_collecting and sample_count < self.config.max_samples:
            try:
                await asyncio.sleep(self.config.interval)
                metrics = await self._get_memory_metrics()
                await self._store_metrics(metrics)

                if await self._should_alert(metrics):
                    await self._send_alert(metrics)

                sample_count += 1

            except Exception as e:
                self.logger.error(f"Error in async memory collection: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _collect_metrics_sync(self) -> None:
        """Synchronous memory metrics collection"""
        sample_count = 0

        while self.is_collecting and sample_count < self.config.max_samples:
            try:
                time.sleep(self.config.interval)
                metrics = self._get_memory_metrics_sync()
                self._store_metrics_sync(metrics)

                if self._should_alert_sync(metrics):
                    self._send_alert_sync(metrics)

                sample_count += 1

            except Exception as e:
                self.logger.error(f"Error in sync memory collection: {e}", exc_info=True)
                time.sleep(1)

    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get current memory metrics"""
        process_memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        metrics = {
            'timestamp': time.time(),
            'rss': process_memory.rss / (1024 * 1024),  # MB
            'vms': process_memory.vms / (1024 * 1024),  # MB
            'system_total': system_memory.total / (1024 * 1024),  # MB
            'system_available': system_memory.available / (1024 * 1024),  # MB
            'system_percent': system_memory.percent,
            'process_percent': self.process.memory_percent()
        }

        if self.config.detailed_metrics:
            metrics.update(await self._get_detailed_memory_metrics())

        return metrics

    def _get_memory_metrics_sync(self) -> Dict[str, Any]:
        """
        Get current memory metrics for synchronous operation.

        Returns:
            Dict[str, Any]: Memory metrics data point
        """
        process_memory = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        metrics = {
            'timestamp': time.time(),
            'rss': process_memory.rss / (1024 * 1024),  # MB
            'vms': process_memory.vms / (1024 * 1024),  # MB
            'system_total': system_memory.total / (1024 * 1024),  # MB
            'system_available': system_memory.available / (1024 * 1024),  # MB
            'system_percent': system_memory.percent,
            'process_percent': self.process.memory_percent()
        }

        if self.config.detailed_metrics:
            try:
                memory_maps = self.process.memory_maps(grouped=True)
                metrics.update({
                    'shared': sum(m.shared for m in memory_maps) / (1024 * 1024),  # MB
                    'private': sum(m.private for m in memory_maps) / (1024 * 1024),  # MB
                    'swap': sum(m.swap for m in memory_maps) / (1024 * 1024),  # MB
                })
            except Exception as e:
                self.logger.warning(f"Could not collect detailed memory metrics: {e}")

        return metrics

    def _should_alert_sync(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics should trigger an alert in synchronous mode.

        Args:
            metrics: Current metrics data point

        Returns:
            bool: True if alert threshold exceeded
        """
        if not self.config.alert_enabled:
            return False

        return (
                metrics['rss'] > self.config.memory_threshold_mb or
                metrics['system_percent'] > 90  # System memory usage above 90%
        )

    def _send_alert_sync(self, metrics: Dict[str, Any]) -> None:
        """
        Send alert for concerning metrics in synchronous mode.

        Args:
            metrics: Metrics that triggered the alert
        """
        alert_message = (
            f"Memory Alert: Process memory usage {metrics['rss']:.2f} MB "
            f"exceeds threshold {self.config.memory_threshold_mb} MB"
        )
        self.logger.warning(alert_message)
        # Additional alert mechanisms could be added here

    async def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory metrics report"""
        if not self.metrics:
            return self._generate_empty_report()

        total_metrics = self._calculate_total_metrics()
        peak_metrics = self._calculate_peak_metrics()
        average_metrics = self._calculate_average_metrics()

        report = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': time.time() - self.start_time,
            'total_samples': len(self.metrics),
            'total_metrics': total_metrics,
            'peak_metrics': peak_metrics,
            'average_metrics': average_metrics,
            'memory_statistics': self._calculate_memory_statistics()
        }

        # Save report to disk
        report_path = self.storage_path / f"memory_report_{int(self.start_time)}.json"
        async with aiofiles.open(report_path, 'w') as f:
            await f.write(json.dumps(report, indent=2))

        return report

    def _generate_empty_report(self) -> Dict[str, Any]:
        """
        Generate empty report when no metrics were collected.

        Returns:
            Dict[str, Any]: Empty report structure with metadata
        """
        return {
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': time.time() - (self.start_time or time.time()),
            'total_samples': 0,
            'error': 'No metrics collected during the session',
            'statistics': {
                'peak_memory_mb': 0,
                'average_memory_mb': 0,
                'peak_system_percent': 0
            }
        }

    def _calculate_total_metrics(self) -> Dict[str, Any]:
        """
        Calculate total metrics from collected data points.

        Returns:
            Dict[str, Any]: Aggregated total metrics
        """
        return {
            key: sum(m[key] for m in self.metrics if key != 'timestamp' and key in m)
            for key in ['rss', 'vms', 'shared', 'private', 'swap']
            if any(key in m for m in self.metrics)
        }

    def _calculate_peak_metrics(self) -> Dict[str, Any]:
        """
        Calculate peak values for each metric.

        Returns:
            Dict[str, Any]: Peak values for all metrics
        """
        # Create a set of all possible metric keys except timestamp
        all_keys = set()
        for m in self.metrics:
            all_keys.update(k for k in m.keys() if k != 'timestamp')

        # Calculate peak values for each metric
        peaks = {}
        for key in all_keys:
            try:
                peaks[key] = max(m.get(key, 0) for m in self.metrics)
            except (ValueError, TypeError):
                # Skip metrics that can't be compared numerically
                continue

        return peaks

    def _calculate_average_metrics(self) -> Dict[str, Any]:
        """
        Calculate average values for each metric.

        Returns:
            Dict[str, Any]: Average values for all metrics
        """
        # Create a set of all possible metric keys except timestamp
        all_keys = set()
        for m in self.metrics:
            all_keys.update(k for k in m.keys() if k != 'timestamp')

        # Calculate averages for each metric
        averages = {}
        for key in all_keys:
            try:
                values = [m.get(key, 0) for m in self.metrics if key in m]
                if values:
                    averages[key] = sum(values) / len(values)
                else:
                    averages[key] = 0
            except (ValueError, TypeError):
                # Skip metrics that can't be averaged
                continue

        return averages

class ProfilingManager:
    """Main class for managing both network and memory profiling"""

    def __init__(self, config: ProfilingConfig):
        self.config = config
        self.network_metrics = NetworkMetrics(config)
        self.memory_metrics = MemoryMetrics(config)
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the profiling manager"""
        log_path = Path(self.config.log_directory) / "profiling_manager.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        if self.config.enable_disk_logging:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if self.config.enable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.DEBUG)

    async def start_profiling(self, profiling_type: ProfilingType = ProfilingType.BOTH) -> None:
        """Start profiling based on specified type"""
        try:
            if profiling_type in (ProfilingType.NETWORK, ProfilingType.BOTH):
                await self.network_metrics.start()

            if profiling_type in (ProfilingType.MEMORY, ProfilingType.BOTH):
                await self.memory_metrics.start()

            self.logger.info(f"Started profiling with type: {profiling_type}")

        except Exception as e:
            self.logger.error(f"Error starting profiling: {e}", exc_info=True)
            await self.stop_profiling()
            raise ProfilingError(f"Failed to start profiling: {str(e)}")

    async def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return combined results"""
        results = {}

        try:
            if self.network_metrics.is_collecting:
                results['network'] = await self.network_metrics.stop()

            if self.memory_metrics.is_collecting:
                results['memory'] = await self.memory_metrics.stop()

            self.logger.info("Stopped profiling successfully")

            # Generate combined report
            await self._generate_combined_report(results)
            return results

        except Exception as e:
            self.logger.error(f"Error stopping profiling: {e}", exc_info=True)
            raise ProfilingError(f"Failed to stop profiling: {str(e)}")

    async def _generate_combined_report(self, results: Dict[str, Any]) -> None:
        """Generate a combined report with both network and memory metrics"""
        report_path = Path(self.config.log_directory) / "combined_reports"
        report_path.mkdir(parents=True, exist_ok=True)

        combined_report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'results': results,
            'correlations': await self._calculate_correlations(results)
        }

        filename = report_path / f"combined_report_{int(time.time())}.json"
        async with aiofiles.open(filename, 'w') as f:
            await f.write(json.dumps(combined_report, indent=2))

    async def _calculate_correlations(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate correlations between network and memory metrics"""
        if 'network' not in results or 'memory' not in results:
            return {}

        try:
            network_timestamps = [m['timestamp'] for m in self.network_metrics.metrics]
            memory_timestamps = [m['timestamp'] for m in self.memory_metrics.metrics]

            # Align metrics by timestamp
            aligned_metrics = self._align_metrics_by_time(
                self.network_metrics.metrics,
                self.memory_metrics.metrics
            )

            correlations = {
                'bandwidth_vs_memory': np.corrcoef(
                    [m['bandwidth_mbps'] for m in aligned_metrics['network']],
                    [m['rss'] for m in aligned_metrics['memory']]
                )[0, 1],
                'packets_vs_memory': np.corrcoef(
                    [m['packets_sent'] + m['packets_recv'] for m in aligned_metrics['network']],
                    [m['rss'] for m in aligned_metrics['memory']]
                )[0, 1]
            }

            return correlations

        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}", exc_info=True)
            return {}

    def _align_metrics_by_time(
            self,
            network_metrics: List[Dict[str, Any]],
            memory_metrics: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Align network and memory metrics by timestamp"""
        aligned_network = []
        aligned_memory = []

        network_idx = 0
        memory_idx = 0

        while network_idx < len(network_metrics) and memory_idx < len(memory_metrics):
            network_time = network_metrics[network_idx]['timestamp']
            memory_time = memory_metrics[memory_idx]['timestamp']

            if abs(network_time - memory_time) < self.config.interval:
                aligned_network.append(network_metrics[network_idx])
                aligned_memory.append(memory_metrics[memory_idx])
                network_idx += 1
                memory_idx += 1
            elif network_time < memory_time:
                network_idx += 1
            else:
                memory_idx += 1

        return {
            'network': aligned_network,
            'memory': aligned_memory
        }

    @asynccontextmanager
    async def profile(self, profiling_type: ProfilingType = ProfilingType.BOTH):
        """Context manager for easy profiling"""
        try:
            await self.start_profiling(profiling_type)
            yield self
        finally:
            await self.stop_profiling()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics from both network and memory profilers"""
        metrics = {}

        if self.network_metrics.is_collecting:
            metrics['network'] = self.network_metrics.metrics[-1] if self.network_metrics.metrics else {}

        if self.memory_metrics.is_collecting:
            metrics['memory'] = self.memory_metrics.metrics[-1] if self.memory_metrics.metrics else {}

        return metrics

    async def cleanup_old_data(self) -> None:
        """Clean up old metric files based on retention policy"""
        if not self.config.backup_enabled:
            return

        current_time = time.time()
        retention_seconds = self.config.retention_days * 86400

        for path in [self.network_metrics.storage_path, self.memory_metrics.storage_path]:
            for file_path in path.glob("**/*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > retention_seconds:
                        try:
                            file_path.unlink()
                            self.logger.info(f"Deleted old file: {file_path}")
                        except Exception as e:
                            self.logger.error(f"Error deleting old file {file_path}: {e}")

    async def backup_data(self, backup_dir: str) -> None:
        """Backup collected data to specified directory"""
        if not self.config.backup_enabled:
            return

        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Create a compressed archive of all data
            import shutil

            source_dirs = [
                self.network_metrics.storage_path,
                self.memory_metrics.storage_path
            ]

            archive_name = f"profiling_backup_{timestamp}.tar.gz"
            archive_path = backup_path / archive_name

            with tarfile.open(archive_path, "w:gz") as tar:
                for source_dir in source_dirs:
                    tar.add(source_dir, arcname=source_dir.name)

            self.logger.info(f"Backup created successfully: {archive_path}")

        except Exception as e:
            self.logger.error(f"Error creating backup: {e}", exc_info=True)
            raise ProfilingError(f"Failed to create backup: {str(e)}")

    def __repr__(self) -> str:
        return (
            f"ProfilingManager(network_active={self.network_metrics.is_collecting}, "
            f"memory_active={self.memory_metrics.is_collecting})"
        )

# Usage Example
async def main():
    config = ProfilingConfig(
        interval=0.1,
        detailed=True,
        log_directory="profiling_logs",
        max_samples=1000,
        enable_disk_logging=True,
        enable_console_output=True,
        async_mode=True,
        memory_threshold_mb=1000.0,
        network_threshold_mbps=100.0
    )

    profiler = ProfilingManager(config)

    async with profiler.profile(ProfilingType.BOTH) as p:
        # Your application code here
        for _ in range(10):
            metrics = await p.get_current_metrics()
            print(f"Current Metrics: {metrics}")
            await asyncio.sleep(1)

    # Cleanup old data
    await profiler.cleanup_old_data()

    # Backup data
    await profiler.backup_data("backup_directory")

if __name__ == "__main__":
    asyncio.run(main())
