"""
Memoraith Network Profiler Module - Enterprise-Grade Implementation
Filename: memoraith/data_collection/network_profiler.py

This module provides comprehensive network traffic monitoring capabilities for deep learning
model profiling, including bandwidth tracking, anomaly detection, and performance optimization.
"""

import logging
import asyncio
import psutil
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from collections import deque
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import threading

# Constants for performance optimization
MAX_SAMPLE_RETENTION = 10000  # Maximum samples to retain in memory
ALERT_THRESHOLD_MBPS = 100.0  # Network bandwidth alert threshold (Mbps)
ANOMALY_Z_SCORE = 3.0  # Z-score threshold for anomaly detection
COLLECTION_JITTER_MS = 2.0  # Collection timing jitter compensation (ms)

@dataclass
class NetworkMetricsConfig:
    """Configuration for network metrics collection."""
    interval: float = 0.1  # Collection interval in seconds
    log_directory: str = "network_logs"
    enable_disk_logging: bool = True
    enable_alerts: bool = True
    alert_threshold_mbps: float = ALERT_THRESHOLD_MBPS
    compression_enabled: bool = True
    max_samples: int = MAX_SAMPLE_RETENTION
    anomaly_detection: bool = True
    anomaly_z_score: float = ANOMALY_Z_SCORE
    detailed_metrics: bool = True
    backup_interval_hours: float = 24.0
    retention_days: int = 7

class AlertSeverity(Enum):
    """Alert severity levels for network events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class NetworkAlert:
    """Network alert data structure."""
    timestamp: float
    metric_type: str
    value: float
    threshold: float
    severity: AlertSeverity
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)

class NetworkProfiler:
    """
    Enterprise-grade network traffic profiler for deep learning model operations.

    Features:
    - High-precision bandwidth measurement
    - Anomaly detection with configurable thresholds
    - Real-time and historical metrics collection
    - Thread-safe asynchronous operation
    - Efficient memory management with circular buffers
    - Comprehensive error handling and recovery
    - Detailed logging with multiple severity levels
    """

    def __init__(self, config: Optional[NetworkMetricsConfig] = None):
        """
        Initialize the network profiler with configuration.

        Args:
            config: Configuration parameters, uses defaults if None
        """
        self.config = config or NetworkMetricsConfig()

        # Core logging setup
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Metrics storage - using deque for memory efficiency
        self._metrics = deque(maxlen=self.config.max_samples)
        self._alerts: List[NetworkAlert] = []
        self._anomalies: List[Dict[str, Any]] = []

        # State management
        self._is_collecting = False
        self._collection_start_time: Optional[float] = None
        self._baseline_counters: Optional[Any] = None
        self._last_counters: Optional[Any] = None
        self._total_bytes_sent = 0
        self._total_bytes_received = 0
        self._peak_bandwidth = 0.0

        # Thread/async safety
        self._lock = asyncio.Lock()
        self._thread_lock = threading.Lock()
        self._task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None

        # Statistics for anomaly detection
        self._bandwidth_values: List[float] = []
        self._bandwidth_mean = 0.0
        self._bandwidth_std = 0.1  # Non-zero default to avoid division by zero

        # Setup storage
        self._storage_path = Path(self.config.log_directory) / "network_metrics"
        self._create_storage_directories()

        self.logger.info("NetworkProfiler initialized with interval: %.3fs", self.config.interval)

    def _setup_logging(self) -> None:
        """Configure logging with appropriate handlers."""
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            if self.config.enable_disk_logging:
                log_dir = Path(self.config.log_directory)
                log_dir.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(log_dir / "network_profiler.log")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

            # Always add console handler for visibility during development
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.logger.setLevel(logging.INFO)

    def _create_storage_directories(self) -> None:
        """Create necessary directories for metric storage."""
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Storage directories created at %s", self._storage_path)
        except (PermissionError, OSError) as e:
            self.logger.error("Failed to create storage directories: %s", str(e))
            # Use fallback location
            self._storage_path = Path.home() / ".memoraith" / "network_metrics"
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self.logger.warning("Using fallback storage location: %s", self._storage_path)

    async def start(self) -> None:
        """
        Start network metrics collection with enterprise-grade error handling.

        Raises:
            RuntimeError: If collection is already active
            OSError: If system counters are unavailable
        """
        try:
            async with self._lock:
                if self._is_collecting:
                    self.logger.warning("Network profiling already active")
                    return

                # Initialize counters
                try:
                    self._baseline_counters = psutil.net_io_counters()
                    self._last_counters = self._baseline_counters
                except (psutil.Error, AttributeError) as e:
                    self.logger.error("Failed to initialize network counters: %s", str(e))
                    raise OSError(f"Network counters unavailable: {str(e)}") from e

                # Reset metrics
                self._metrics.clear()
                self._alerts.clear()
                self._anomalies.clear()
                self._bandwidth_values.clear()
                self._total_bytes_sent = 0
                self._total_bytes_received = 0
                self._peak_bandwidth = 0.0

                # Start collection
                self._collection_start_time = time.time()
                self._is_collecting = True

                # Launch tasks
                self._task = asyncio.create_task(self._collection_loop())
                if self.config.compression_enabled:
                    self._backup_task = asyncio.create_task(self._periodic_backup())

                await self._save_state("started")
                self.logger.info("Network profiling started at %.3f", self._collection_start_time)

        except Exception as e:
            self.logger.error("Failed to start network profiling: %s", str(e), exc_info=True)
            # Ensure cleanup on failure
            self._is_collecting = False
            if self._task:
                self._task.cancel()
                self._task = None
            if self._backup_task:
                self._backup_task.cancel()
                self._backup_task = None
            raise

    async def _save_state(self, state: str) -> None:
        """
        Save profiler state to disk for monitoring and recovery.

        Args:
            state: Current operational state
        """
        try:
            state_file = self._storage_path / "profiler_state.json"
            state_data = {
                "timestamp": time.time(),
                "state": state,
                "collection_start_time": self._collection_start_time,
                "samples_collected": len(self._metrics),
                "alerts_triggered": len(self._alerts),
                "anomalies_detected": len(self._anomalies),
                "config": {k: str(v) if isinstance(v, Path) else v
                           for k, v in self.config.__dict__.items()}
            }

            # Atomic write using temporary file
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)

            # Atomic rename
            temp_file.replace(state_file)

        except (OSError, IOError) as e:
            self.logger.warning("Failed to save state: %s", str(e))

    async def stop(self) -> Dict[str, Any]:
        """
        Stop network metrics collection and return comprehensive results.

        Returns:
            Dict[str, Any]: Complete analysis results

        Raises:
            RuntimeError: If collection is not active
        """
        if not self._is_collecting:
            self.logger.warning("Network profiling not active")
            return {"error": "Not active", "metrics_count": 0}

        try:
            async with self._lock:
                self._is_collecting = False

                # Cancel collection tasks
                if self._task:
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        pass
                    self._task = None

                if self._backup_task:
                    self._backup_task.cancel()
                    try:
                        await self._backup_task
                    except asyncio.CancelledError:
                        pass
                    self._backup_task = None

                # Generate final report
                results = await self._generate_report()

                # Save final state
                await self._save_state("stopped")

                self.logger.info("Network profiling stopped. Collected %d samples", len(self._metrics))
                return results

        except Exception as e:
            self.logger.error("Error stopping network profiling: %s", str(e), exc_info=True)
            # Ensure we mark as stopped even on error
            self._is_collecting = False
            raise

    async def _collection_loop(self) -> None:
        """
        Main collection loop with adaptive timing and error recovery.
        """
        adjustment_factor = 0.0  # Timing adjustment for interval accuracy
        sample_count = 0
        error_count = 0

        while self._is_collecting:
            loop_start = time.time()

            try:
                # Retrieve system network counters
                current_counters = psutil.net_io_counters()

                # Calculate metrics
                metrics = self._calculate_metrics(self._last_counters, current_counters)

                # Update statistics for anomaly detection
                self._update_statistics(metrics)

                # Check for alerts and anomalies
                if await self._check_for_alerts(metrics):
                    await self._handle_alert(metrics)

                if self.config.anomaly_detection and await self._check_for_anomalies(metrics):
                    await self._handle_anomaly(metrics)

                # Store metrics
                await self._store_metrics(metrics)

                # Update tracking variables
                self._last_counters = current_counters
                self._total_bytes_sent += metrics['bytes_sent']
                self._total_bytes_received += metrics['bytes_recv']
                self._peak_bandwidth = max(self._peak_bandwidth, metrics['bandwidth_mbps'])

                sample_count += 1

                # Adaptive timing to maintain consistent interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config.interval - loop_duration + adjustment_factor)

                # Correct for timing drift every 10 samples
                if sample_count % 10 == 0:
                    actual_interval = (time.time() - self._collection_start_time) / sample_count
                    drift = actual_interval - self.config.interval
                    adjustment_factor = -drift * 0.1  # Gradual adjustment

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                self.logger.debug("Collection loop cancelled")
                break

            except Exception as e:
                error_count += 1
                self.logger.error("Error in network collection loop: %s", str(e))

                # Exponential backoff for persistent errors
                if error_count > 3:
                    await asyncio.sleep(min(30, 2 ** (error_count - 3)))
                else:
                    await asyncio.sleep(0.5)  # Short delay for transient errors

                # Reset after too many errors to avoid resource exhaustion
                if error_count > 10:
                    self.logger.critical("Too many errors in collection loop, resetting counters")
                    try:
                        self._last_counters = psutil.net_io_counters()
                        error_count = 0  # Reset error count after recovery
                    except Exception:
                        pass  # Silently continue if reset fails

    def _calculate_metrics(self, last_counters: Any, current_counters: Any) -> Dict[str, Any]:
        """
        Calculate comprehensive network metrics between counter states.

        Args:
            last_counters: Previous network counter state
            current_counters: Current network counter state

        Returns:
            Dict[str, Any]: Calculated metrics
        """
        # Calculate time delta for accurate rate calculations
        current_time = time.time()
        time_delta = (current_time - self._collection_start_time if len(self._metrics) == 0
                      else current_time - self._metrics[-1]['timestamp'] if self._metrics
        else self.config.interval)

        # Ensure we don't divide by zero
        if time_delta < 0.001:
            time_delta = 0.001

        # Basic metrics calculation
        bytes_sent = current_counters.bytes_sent - last_counters.bytes_sent
        bytes_recv = current_counters.bytes_recv - last_counters.bytes_recv
        packets_sent = current_counters.packets_sent - last_counters.packets_sent
        packets_recv = current_counters.packets_recv - last_counters.packets_recv

        # Handle counter rollover (unsigned integers)
        if bytes_sent < 0:
            bytes_sent = current_counters.bytes_sent
        if bytes_recv < 0:
            bytes_recv = current_counters.bytes_recv
        if packets_sent < 0:
            packets_sent = current_counters.packets_sent
        if packets_recv < 0:
            packets_recv = current_counters.packets_recv

        # Calculate bandwidth in Mbps (megabits per second)
        bandwidth_mbps = ((bytes_sent + bytes_recv) * 8) / (time_delta * 1024 * 1024)

        metrics = {
            'timestamp': current_time,
            'bytes_sent': bytes_sent,
            'bytes_recv': bytes_recv,
            'packets_sent': packets_sent,
            'packets_recv': packets_recv,
            'bandwidth_mbps': bandwidth_mbps,
            'packet_rate': (packets_sent + packets_recv) / time_delta
        }

        # Add detailed metrics if configured
        if self.config.detailed_metrics:
            errors_in = current_counters.errin - last_counters.errin
            errors_out = current_counters.errout - last_counters.errout
            drops_in = current_counters.dropin - last_counters.dropin
            drops_out = current_counters.dropout - last_counters.dropout

            # Handle counter rollover
            if errors_in < 0: errors_in = current_counters.errin
            if errors_out < 0: errors_out = current_counters.errout
            if drops_in < 0: drops_in = current_counters.dropin
            if drops_out < 0: drops_out = current_counters.dropout

            # Calculate error rates
            total_packets = packets_sent + packets_recv
            if total_packets > 0:
                error_rate = (errors_in + errors_out + drops_in + drops_out) / total_packets * 100
            else:
                error_rate = 0.0

            metrics.update({
                'errin': errors_in,
                'errout': errors_out,
                'dropin': drops_in,
                'dropout': drops_out,
                'error_rate': error_rate,
                'average_packet_size': ((bytes_sent + bytes_recv) / total_packets) if total_packets > 0 else 0,
                'bytes_sent_rate': bytes_sent / time_delta,
                'bytes_recv_rate': bytes_recv / time_delta
            })

        return metrics

    def _update_statistics(self, metrics: Dict[str, Any]) -> None:
        """
        Update running statistics for anomaly detection.

        Args:
            metrics: Current metrics data point
        """
        # Add to values list
        self._bandwidth_values.append(metrics['bandwidth_mbps'])

        # Limit size for memory efficiency
        if len(self._bandwidth_values) > 1000:
            # Remove oldest values while keeping distribution
            self._bandwidth_values = self._bandwidth_values[-1000:]

        # Only recalculate periodically to improve performance
        if len(self._bandwidth_values) % 5 == 0 and len(self._bandwidth_values) >= 2:
            self._bandwidth_mean = np.mean(self._bandwidth_values)
            self._bandwidth_std = max(0.1, np.std(self._bandwidth_values))  # Ensure non-zero

    async def _check_for_alerts(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if current metrics should trigger an alert.

        Args:
            metrics: Current metrics data point

        Returns:
            bool: True if alert threshold exceeded
        """
        if not self.config.enable_alerts:
            return False

        # Check primary bandwidth threshold
        if metrics['bandwidth_mbps'] > self.config.alert_threshold_mbps:
            return True

        # Check error rate if detailed metrics enabled
        if self.config.detailed_metrics and metrics.get('error_rate', 0) > 5.0:  # 5% error rate
            return True

        return False

    async def _handle_alert(self, metrics: Dict[str, Any]) -> None:
        """
        Process and record network alert.

        Args:
            metrics: Metrics that triggered the alert
        """
        # Determine severity
        severity = AlertSeverity.WARNING
        if metrics['bandwidth_mbps'] > self.config.alert_threshold_mbps * 2:
            severity = AlertSeverity.CRITICAL

        # Create alert object
        alert = NetworkAlert(
            timestamp=metrics['timestamp'],
            metric_type='bandwidth',
            value=metrics['bandwidth_mbps'],
            threshold=self.config.alert_threshold_mbps,
            severity=severity,
            message=f"Network bandwidth {metrics['bandwidth_mbps']:.2f} Mbps exceeds threshold {self.config.alert_threshold_mbps} Mbps",
            metrics=metrics.copy()
        )

        # Log alert
        log_method = self.logger.warning if severity == AlertSeverity.WARNING else self.logger.error
        log_method(alert.message)

        # Store alert
        async with self._lock:
            self._alerts.append(alert)

        # Write to alert log
        try:
            alert_file = self._storage_path / "alerts.log"
            with open(alert_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))} - "
                        f"{alert.severity.value.upper()} - {alert.message}\n")
        except (OSError, IOError) as e:
            self.logger.warning("Failed to write alert to log: %s", str(e))

    async def _check_for_anomalies(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if current metrics represent an anomaly.

        Args:
            metrics: Current metrics data point

        Returns:
            bool: True if anomaly detected
        """
        # Need sufficient baseline data
        if len(self._bandwidth_values) < 10:
            return False

        # Calculate z-score for bandwidth
        z_score = abs((metrics['bandwidth_mbps'] - self._bandwidth_mean) / self._bandwidth_std)

        return z_score > self.config.anomaly_z_score

    async def _handle_anomaly(self, metrics: Dict[str, Any]) -> None:
        """
        Process and record network anomaly.

        Args:
            metrics: Metrics that represent an anomaly
        """
        # Calculate z-score
        z_score = (metrics['bandwidth_mbps'] - self._bandwidth_mean) / self._bandwidth_std

        # Create anomaly record
        anomaly = {
            'timestamp': metrics['timestamp'],
            'metric_type': 'bandwidth',
            'value': metrics['bandwidth_mbps'],
            'expected_value': self._bandwidth_mean,
            'z_score': z_score,
            'severity': 'high' if abs(z_score) > 5 else 'medium',
            'message': f"Network anomaly detected: bandwidth {metrics['bandwidth_mbps']:.2f} Mbps (z-score: {z_score:.2f})"
        }

        # Log anomaly
        self.logger.warning(anomaly['message'])

        # Store anomaly
        async with self._lock:
            self._anomalies.append(anomaly)

    async def _store_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Store metrics with thread-safe operation and optional disk persistence.

        Args:
            metrics: Metrics data point to store
        """
        # Store in memory
        async with self._lock:
            self._metrics.append(metrics)

        # Write to disk if enabled
        if self.config.enable_disk_logging:
            await self._write_metrics_to_disk(metrics)

    async def _write_metrics_to_disk(self, metrics: Dict[str, Any]) -> None:
        """
        Write metrics to disk with error handling and retry.

        Args:
            metrics: Metrics data to write
        """
        if not self._collection_start_time:
            return

        try:
            # Batch metrics by hour for efficient storage
            hour_timestamp = int(self._collection_start_time / 3600) * 3600
            metrics_file = self._storage_path / f"network_metrics_{hour_timestamp}.jsonl"

            # Using append mode for continuous logging
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')

        except (OSError, IOError) as e:
            self.logger.warning("Failed to write metrics to disk: %s", str(e))

            # Retry with fallback location on permission errors
            if isinstance(e, PermissionError):
                try:
                    fallback_path = Path.home() / ".memoraith" / "network_metrics"
                    fallback_path.mkdir(parents=True, exist_ok=True)

                    hour_timestamp = int(self._collection_start_time / 3600) * 3600
                    metrics_file = fallback_path / f"network_metrics_{hour_timestamp}.jsonl"

                    with open(metrics_file, 'a') as f:
                        f.write(json.dumps(metrics) + '\n')

                    # Update storage path to fallback
                    self._storage_path = fallback_path
                    self.logger.info("Using fallback storage location: %s", fallback_path)

                except Exception as fallback_error:
                    self.logger.error("Fallback storage failed: %s", str(fallback_error))

    async def _periodic_backup(self) -> None:
        """Periodically compress and backup old metric files."""
        import gzip
        import shutil
        from datetime import datetime, timedelta

        while self._is_collecting:
            try:
                # Wait for next backup interval
                await asyncio.sleep(self.config.backup_interval_hours * 3600)

                # Find files older than 1 day
                now = time.time()
                files_to_compress = []

                for file_path in self._storage_path.glob("*.jsonl"):
                    if now - file_path.stat().st_mtime > 86400:  # 24 hours
                        files_to_compress.append(file_path)

                # Compress files
                for file_path in files_to_compress:
                    gz_path = file_path.with_suffix('.jsonl.gz')

                    if not gz_path.exists():
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(gz_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)

                        # Verify compression worked before deleting original
                        if gz_path.exists() and gz_path.stat().st_size > 0:
                            file_path.unlink()
                            self.logger.debug("Compressed %s", file_path.name)

                # Clean up old files based on retention policy
                retention_threshold = datetime.now() - timedelta(days=self.config.retention_days)
                threshold_timestamp = retention_threshold.timestamp()

                for file_path in self._storage_path.glob("*.gz"):
                    try:
                        # Extract timestamp from filename
                        filename = file_path.stem
                        if filename.startswith("network_metrics_"):
                            try:
                                timestamp = int(filename.split("_")[-1])
                                if timestamp < threshold_timestamp:
                                    file_path.unlink()
                                    self.logger.debug("Deleted old file %s", file_path.name)
                            except (ValueError, IndexError):
                                # Couldn't parse timestamp, use file modification time as fallback
                                if file_path.stat().st_mtime < threshold_timestamp:
                                    file_path.unlink()
                    except OSError as e:
                        self.logger.warning("Error cleaning up file %s: %s", file_path.name, str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in backup task: %s", str(e))
                await asyncio.sleep(3600)  # Wait an hour and retry

    async def _generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report with statistical analysis.

        Returns:
            Dict[str, Any]: Complete profiling report
        """
        if not self._metrics:
            return self._generate_empty_report()

        try:
            # Convert deque to list for analysis
            metrics_list = list(self._metrics)

            # Extract key metrics
            bandwidth_values = [m['bandwidth_mbps'] for m in metrics_list]

            # Basic statistics
            statistics = {
                'count': len(metrics_list),
                'duration': metrics_list[-1]['timestamp'] - self._collection_start_time,
                'total_bytes_sent': self._total_bytes_sent,
                'total_bytes_received': self._total_bytes_received,
                'peak_bandwidth_mbps': self._peak_bandwidth,
                'average_bandwidth_mbps': sum(bandwidth_values) / len(bandwidth_values),
                'median_bandwidth_mbps': sorted(bandwidth_values)[len(bandwidth_values) // 2],
                'std_bandwidth_mbps': np.std(bandwidth_values),
                'alerts_triggered': len(self._alerts),
                'anomalies_detected': len(self._anomalies)
            }

            # Add percentile analysis
            if len(bandwidth_values) >= 10:
                statistics.update({
                    'bandwidth_p95_mbps': np.percentile(bandwidth_values, 95),
                    'bandwidth_p99_mbps': np.percentile(bandwidth_values, 99),
                    'bandwidth_p5_mbps': np.percentile(bandwidth_values, 5)
                })

            # Add detailed error analysis if available
            if self.config.detailed_metrics and any('error_rate' in m for m in metrics_list):
                error_rates = [m.get('error_rate', 0) for m in metrics_list]
                statistics.update({
                    'peak_error_rate': max(error_rates),
                    'average_error_rate': sum(error_rates) / len(error_rates),
                    'total_errors': sum(m.get('errin', 0) + m.get('errout', 0) +
                                        m.get('dropin', 0) + m.get('dropout', 0)
                                        for m in metrics_list)
                })

            # Prepare comprehensive report
            report = {
                'start_time': self._collection_start_time,
                'end_time': time.time(),
                'statistics': statistics,
                'anomalies': self._anomalies.copy(),
                'alerts': [a.__dict__ for a in self._alerts],
                'sampling_interval': self.config.interval,
                'storage_location': str(self._storage_path),
                'config': {k: str(v) if isinstance(v, Path) else v
                           for k, v in self.config.__dict__.items()},
                'summary': f"Collected {len(metrics_list)} samples over {statistics['duration']:.2f} seconds. "
                           f"Peak bandwidth: {statistics['peak_bandwidth_mbps']:.2f} Mbps. "
                           f"Average bandwidth: {statistics['average_bandwidth_mbps']:.2f} Mbps."
            }

            # Save report to disk
            await self._save_report(report)

            return report

        except Exception as e:
            self.logger.error("Error generating report: %s", str(e), exc_info=True)
            return {
                'error': f"Failed to generate report: {str(e)}",
                'metrics_count': len(self._metrics),
                'start_time': self._collection_start_time,
                'end_time': time.time()
            }

    def _generate_empty_report(self) -> Dict[str, Any]:
        """
        Generate report when no metrics were collected.

        Returns:
            Dict[str, Any]: Empty report structure
        """
        return {
            'start_time': self._collection_start_time,
            'end_time': time.time(),
            'duration': time.time() - (self._collection_start_time or time.time()),
            'error': 'No metrics collected during the session',
            'statistics': {
                'count': 0,
                'total_bytes_sent': 0,
                'total_bytes_received': 0,
                'peak_bandwidth_mbps': 0.0,
                'average_bandwidth_mbps': 0.0
            }
        }
    async def _save_report(self, report: Dict[str, Any]) -> None:
        """
        Save comprehensive report to disk with atomicity guarantees.

        Args:
            report: Complete analysis report to persist

        Note:
            Uses atomic write pattern to prevent data corruption during writes
        """
        try:
            # Generate timestamped filename
            report_file = self._storage_path / f"network_report_{int(self._collection_start_time or time.time())}.json"
            temp_file = report_file.with_suffix('.tmp')

            # Write to temporary file first for atomic operation
            with open(temp_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Atomic rename to final filename
            temp_file.replace(report_file)

            self.logger.info("Network profiling report saved to %s", report_file)

        except Exception as e:
            self.logger.error("Failed to save report: %s", str(e))

        # Synchronous API methods for compatibility with different execution environments

    def _should_alert_sync(self, metrics: Dict[str, Any]) -> bool:
        """
        Synchronous version of alert threshold checking.

        Args:
            metrics: Current metrics data point

        Returns:
            bool: True if metrics exceed alert thresholds
        """
        if not self.config.enable_alerts:
            return False

        # Primary bandwidth check
        if metrics['bandwidth_mbps'] > self.config.alert_threshold_mbps:
            return True

        # Error rate check (if detailed metrics available)
        if self.config.detailed_metrics and metrics.get('error_rate', 0) > 5.0:
            return True

        return False

    def _send_alert_sync(self, metrics: Dict[str, Any]) -> None:
        """
        Synchronous version of alert handling for non-async contexts.

        Args:
            metrics: Metrics that triggered the alert
        """
        with self._thread_lock:
            # Determine severity
            severity = "CRITICAL" if metrics['bandwidth_mbps'] > self.config.alert_threshold_mbps * 2 else "WARNING"

            # Log alert
            message = f"Network bandwidth {metrics['bandwidth_mbps']:.2f} Mbps exceeds threshold {self.config.alert_threshold_mbps} Mbps"
            if severity == "CRITICAL":
                self.logger.error(message)
            else:
                self.logger.warning(message)

            # Store alert (best effort)
            try:
                alert_file = self._storage_path / "alerts.log"
                with open(alert_file, 'a') as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {severity} - {message}\n")
            except (OSError, IOError):
                pass  # Best effort, don't raise in sync context

    async def get_metrics(self) -> List[Dict[str, Any]]:
        """
        Get collected metrics with thread-safe access.

        Returns:
            List[Dict[str, Any]]: Copy of all collected metrics
        """
        async with self._lock:
            return list(self._metrics)

    async def get_latest_metrics(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get most recent metrics for real-time monitoring.

        Args:
            count: Number of most recent metrics to return

        Returns:
            List[Dict[str, Any]]: Most recent metrics
        """
        async with self._lock:
            return list(self._metrics)[-count:]

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all triggered alerts.

        Returns:
            List[Dict[str, Any]]: All recorded alerts
        """
        async with self._lock:
            return [a.__dict__ for a in self._alerts]

    async def get_anomalies(self) -> List[Dict[str, Any]]:
        """
        Get all detected anomalies.

        Returns:
            List[Dict[str, Any]]: All detected anomalies
        """
        async with self._lock:
            return self._anomalies.copy()

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistical summaries without generating full report.

        Returns:
            Dict[str, Any]: Key performance statistics
        """
        if not self._metrics:
            return {
                'count': 0,
                'peak_bandwidth_mbps': 0.0,
                'average_bandwidth_mbps': 0.0,
                'total_bytes': 0
            }

        async with self._lock:
            metrics_list = list(self._metrics)
            bandwidth_values = [m['bandwidth_mbps'] for m in metrics_list]

            return {
                'count': len(metrics_list),
                'duration': metrics_list[-1]['timestamp'] - self._collection_start_time if self._collection_start_time else 0,
                'total_bytes_sent': self._total_bytes_sent,
                'total_bytes_received': self._total_bytes_received,
                'peak_bandwidth_mbps': self._peak_bandwidth,
                'average_bandwidth_mbps': sum(bandwidth_values) / max(1, len(bandwidth_values)),
                'current_bandwidth_mbps': bandwidth_values[-1] if bandwidth_values else 0.0,
                'alerts_triggered': len(self._alerts),
                'anomalies_detected': len(self._anomalies)
            }

    async def clear_data(self) -> None:
        """
        Clear all collected metrics while preserving configuration.
        """
        async with self._lock:
            self._metrics.clear()
            self._alerts.clear()
            self._anomalies.clear()
            self._bandwidth_values.clear()
            self._total_bytes_sent = 0
            self._total_bytes_received = 0
            self._peak_bandwidth = 0.0

        self.logger.info("Network profiling data cleared")

    async def export_metrics(self, format: str = 'json', filename: Optional[str] = None) -> str:
        """
        Export collected metrics to specified format.

        Args:
            format: Export format ('json', 'csv')
            filename: Optional filename, auto-generated if None

        Returns:
            str: Path to exported file

        Raises:
            ValueError: If format is unsupported
            OSError: If export operation fails
        """
        if format not in ('json', 'csv'):
            raise ValueError(f"Unsupported export format: {format}")

        if not filename:
            timestamp = int(time.time())
            filename = f"network_metrics_export_{timestamp}.{format}"

        export_path = self._storage_path / filename

        try:
            async with self._lock:
                metrics_list = list(self._metrics)

            if format == 'json':
                with open(export_path, 'w') as f:
                    json.dump(metrics_list, f, indent=2, default=str)
            else:  # csv
                import csv

                # Determine all possible fields by analyzing metrics
                fields = set()
                for metric in metrics_list:
                    fields.update(metric.keys())

                fields = sorted(list(fields))

                with open(export_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    for metric in metrics_list:
                        writer.writerow(metric)

            self.logger.info("Metrics exported to %s", export_path)
            return str(export_path)

        except Exception as e:
            self.logger.error("Export failed: %s", str(e))
            raise OSError(f"Failed to export metrics: {str(e)}")

    def __del__(self) -> None:
        """
        Clean up resources when object is garbage collected.
        """
        # Cancel any running tasks
        if hasattr(self, '_is_collecting') and self._is_collecting:
            if hasattr(self, '_task') and self._task:
                self._task.cancel()
            if hasattr(self, '_backup_task') and self._backup_task:
                self._backup_task.cancel()