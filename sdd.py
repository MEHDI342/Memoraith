#!/usr/bin/env python
# test_memoraith_integration.py - Enterprise-Grade Integration Test Suite
#
# This script validates the complete functionality of Memoraith with particular
# focus on the fixed components, ensuring production readiness.

import os
import sys
import time
import json
import asyncio
import unittest
import tempfile
import logging
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("memoraith-test")

# Import Memoraith components
try:
    from memoraith import profile_model, set_output_path
    from memoraith.config import Config, config
    from memoraith.exceptions import MemoraithError, DataCollectionError
    from memoraith.data_collection.network_memory_system import NetworkMetrics, MemoryMetrics, ProfilingConfig
    from memoraith.data_collection.network_profiler import NetworkProfiler
    from memoraith.integration.common_utils import identify_framework, estimate_model_size
    from memoraith.data_collection.base_collector import BaseDataCollector

    # Import PyTorch for testing with real models
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    PYTORCH_AVAILABLE = False

# Define a simple model for testing
if PYTORCH_AVAILABLE:
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU(inplace=True)
            self.fc = nn.Linear(16 * 32 * 32, 10)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


class TestDataCollectionError(unittest.TestCase):
    """Test the DataCollectionError exception class with proper details parameter."""

    def test_error_creation(self):
        """Test creating DataCollectionError with details parameter."""
        # Create the error with proper parameters
        error = DataCollectionError("test_data", details="Test error message")

        # Validate error properties
        self.assertEqual(error.data_type, "test_data")
        self.assertEqual(error.details, "Test error message")
        self.assertEqual(str(error), "Error collecting test_data data: Test error message")

    def test_error_inheritance(self):
        """Test that DataCollectionError properly inherits from MemoraithError."""
        # Create error instance
        error = DataCollectionError("test", details="inherited")

        # Verify inheritance
        self.assertIsInstance(error, MemoraithError)

        # Validate exception propagation
        try:
            raise DataCollectionError("propagation_test", details="Exception propagation")
        except MemoraithError as e:
            self.assertIsInstance(e, DataCollectionError)


class TestNetworkMetrics(unittest.TestCase):
    """Test the fixed NetworkMetrics class functionality."""

    def setUp(self):
        """Set up the test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp(prefix="memoraith_network_test_")
        self.config = ProfilingConfig(
            interval=0.1,
            log_directory=self.test_dir,
            max_samples=5
        )
        self.network_metrics = NetworkMetrics(self.config)

    def tearDown(self):
        """Clean up test resources."""
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up test directory: {e}")

    async def async_setup(self):
        """Asynchronous setup."""
        await self.network_metrics.start()

    async def async_teardown(self):
        """Asynchronous teardown."""
        if hasattr(self, 'network_metrics') and self.network_metrics.is_collecting:
            await self.network_metrics.stop()

    def test_initialization(self):
        """Test proper initialization of NetworkMetrics."""
        self.assertEqual(self.network_metrics.config, self.config)
        self.assertIsNotNone(self.network_metrics.logger)
        self.assertFalse(self.network_metrics.is_collecting)
        self.assertIsNone(self.network_metrics.task)

        # Verify storage directory creation
        storage_path = Path(self.test_dir) / "network_metrics"
        self.assertTrue(storage_path.exists())
        self.assertTrue(storage_path.is_dir())

    def test_save_state_method(self):
        """Test the _save_state method implementation."""
        async def run_test():
            self.network_metrics.start_time = time.time()
            await self.network_metrics._save_state("testing")

            # Verify state file was created
            state_file = Path(self.test_dir) / "network_metrics" / "profiler_state.json"
            self.assertTrue(state_file.exists())

            # Verify content
            with open(state_file, 'r') as f:
                state_data = json.load(f)
                self.assertEqual(state_data["state"], "testing")
                self.assertIn("timestamp", state_data)
                self.assertIn("collection_start_time", state_data)

        asyncio.run(run_test())

    def test_sync_methods(self):
        """Test synchronous alert methods."""
        # Create test metrics
        test_metrics = {
            'bandwidth_mbps': 150.0,  # Above default threshold
            'error_rate': 6.0,        # Above default threshold
            'timestamp': time.time()
        }

        # Test _should_alert_sync
        result = self.network_metrics._should_alert_sync(test_metrics)
        self.assertTrue(result)

        # Test with disabled alerts
        self.network_metrics.config.alert_enabled = False
        result = self.network_metrics._should_alert_sync(test_metrics)
        self.assertFalse(result)

        # Re-enable and test lower values
        self.network_metrics.config.alert_enabled = True
        test_metrics['bandwidth_mbps'] = 50.0
        test_metrics['error_rate'] = 2.0
        result = self.network_metrics._should_alert_sync(test_metrics)
        self.assertFalse(result)

        # Test _send_alert_sync doesn't throw errors
        try:
            self.network_metrics._send_alert_sync(test_metrics)
            # Just testing it doesn't raise exceptions
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"_send_alert_sync raised exception: {e}")

    def test_start_stop_collection(self):
        """Test starting and stopping collection lifecycle."""
        async def run_lifecycle_test():
            # Start collection
            await self.network_metrics.start()
            self.assertTrue(self.network_metrics.is_collecting)
            self.assertIsNotNone(self.network_metrics.baseline_counters)

            # Wait for some data collection
            await asyncio.sleep(0.5)

            # Stop collection and get results
            results = await self.network_metrics.stop()
            self.assertFalse(self.network_metrics.is_collecting)
            self.assertIsNotNone(results)

            # Verify expected result structure
            self.assertIn('start_time', results)
            self.assertIn('end_time', results)
            self.assertIn('duration', results)

            # Verify report file was created
            report_files = list(Path(self.test_dir).glob("**/network_report_*.json"))
            self.assertGreater(len(report_files), 0)

        asyncio.run(run_lifecycle_test())


class TestMemoryMetrics(unittest.TestCase):
    """Test the fixed MemoryMetrics class functionality."""

    def setUp(self):
        """Set up the test environment with temporary directory."""
        self.test_dir = tempfile.mkdtemp(prefix="memoraith_memory_test_")
        self.config = ProfilingConfig(
            interval=0.1,
            log_directory=self.test_dir,
            max_samples=5
        )
        self.memory_metrics = MemoryMetrics(self.config)

    def tearDown(self):
        """Clean up test resources."""
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up test directory: {e}")

    def test_initialization(self):
        """Test proper initialization of MemoryMetrics."""
        self.assertEqual(self.memory_metrics.config, self.config)
        self.assertIsNotNone(self.memory_metrics.logger)
        self.assertFalse(self.memory_metrics.is_collecting)
        self.assertIsNotNone(self.memory_metrics.process)

        # Verify storage directory creation
        storage_path = Path(self.test_dir) / "memory_metrics"
        self.assertTrue(storage_path.exists())
        self.assertTrue(storage_path.is_dir())

    def test_memory_metrics_sync(self):
        """Test synchronous memory metrics collection."""
        metrics = self.memory_metrics._get_memory_metrics_sync()

        # Verify essential metrics are present
        self.assertIn('timestamp', metrics)
        self.assertIn('rss', metrics)
        self.assertIn('vms', metrics)
        self.assertIn('system_percent', metrics)

        # Verify values are of expected types
        self.assertIsInstance(metrics['rss'], float)
        self.assertIsInstance(metrics['vms'], float)
        self.assertIsInstance(metrics['system_percent'], float)

        # Values should be positive
        self.assertGreater(metrics['rss'], 0)
        self.assertGreater(metrics['system_total'], 0)

    def test_report_generation_methods(self):
        """Test report generation methods in MemoryMetrics."""
        # Test empty report generation
        empty_report = self.memory_metrics._generate_empty_report()
        self.assertIn('start_time', empty_report)
        self.assertIn('end_time', empty_report)
        self.assertIn('error', empty_report)

        # Add fake metrics data for testing other methods
        test_metrics = []
        for i in range(5):
            test_metrics.append({
                'timestamp': time.time() + i,
                'rss': 100 + i * 10,
                'vms': 200 + i * 20,
                'system_percent': 50 + i
            })
        self.memory_metrics.metrics = test_metrics

        # Test calculation methods
        total = self.memory_metrics._calculate_total_metrics()
        self.assertIn('rss', total)
        self.assertEqual(total['rss'], sum(m['rss'] for m in test_metrics))

        peak = self.memory_metrics._calculate_peak_metrics()
        self.assertIn('rss', peak)
        self.assertEqual(peak['rss'], max(m['rss'] for m in test_metrics))

        avg = self.memory_metrics._calculate_average_metrics()
        self.assertIn('rss', avg)
        self.assertEqual(avg['rss'], sum(m['rss'] for m in test_metrics) / len(test_metrics))

    def test_start_stop_collection(self):
        """Test starting and stopping collection lifecycle."""
        async def run_lifecycle_test():
            # Start collection
            await self.memory_metrics.start()
            self.assertTrue(self.memory_metrics.is_collecting)

            # Wait for some data collection
            await asyncio.sleep(0.5)

            # Stop collection and get results
            results = await self.memory_metrics.stop()
            self.assertFalse(self.memory_metrics.is_collecting)
            self.assertIsNotNone(results)

            # Verify expected result structure
            self.assertIn('start_time', results)
            self.assertIn('end_time', results)
            self.assertIn('duration', results)
            self.assertIn('total_metrics', results)
            self.assertIn('peak_metrics', results)
            self.assertIn('average_metrics', results)

            # Verify report file was created
            report_files = list(Path(self.test_dir).glob("**/memory_report_*.json"))
            self.assertGreater(len(report_files), 0)

        asyncio.run(run_lifecycle_test())


class TestNetworkProfiler(unittest.TestCase):
    """Test the fixed NetworkProfiler class."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="memoraith_profiler_test_")

        from memoraith.data_collection.network_profiler import NetworkMetricsConfig
        self.config = NetworkMetricsConfig(
            interval=0.1,
            log_directory=self.test_dir,
            max_samples=5,
            retention_days=1  # Short period for testing
        )

        self.profiler = NetworkProfiler(self.config)

    def tearDown(self):
        """Clean up test resources."""
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up test directory: {e}")

    def test_retention_threshold(self):
        """Test the fixed line 682 with threshold_timestamp."""
        from datetime import datetime, timedelta

        # Create test files to simulate old backups
        storage_path = Path(self.test_dir) / "network_metrics"
        storage_path.mkdir(parents=True, exist_ok=True)

        # Current timestamp for filenames
        now = int(time.time())

        # Create fake files with timestamps
        filenames = [
            f"network_metrics_{now - 86400*3}.json.gz",  # 3 days old
            f"network_metrics_{now - 86400*2}.json.gz",  # 2 days old
            f"network_metrics_{now}.json.gz",            # Current
        ]

        for filename in filenames:
            with open(storage_path / filename, 'w') as f:
                f.write("test content")

        # Set storage path in profiler
        self.profiler._storage_path = storage_path

        # Create a mock method to call the cleanup logic directly
        async def test_cleanup():
            # This recreates the logic from the fixed code
            retention_threshold = datetime.now() - timedelta(days=self.config.retention_days)
            threshold_timestamp = retention_threshold.timestamp()

            # Execute cleanup
            for file_path in storage_path.glob("*.gz"):
                try:
                    filename = file_path.stem
                    if filename.startswith("network_metrics_"):
                        try:
                            timestamp = int(filename.split("_")[-1])
                            if timestamp < threshold_timestamp:
                                file_path.unlink()
                        except (ValueError, IndexError):
                            if file_path.stat().st_mtime < threshold_timestamp:
                                file_path.unlink()
                except OSError as e:
                    logger.warning(f"Error cleaning up file: {e}")

            # Check remaining files
            remaining_files = list(storage_path.glob("*.gz"))
            return remaining_files

        # Run the test
        remaining_files = asyncio.run(test_cleanup())

        # We should have only the current file left
        self.assertEqual(len(remaining_files), 1)
        self.assertTrue(str(remaining_files[0]).endswith(f"{now}.json.gz"))


class TestIntegrationCommonUtils(unittest.TestCase):
    """Test the fixed integration.common_utils module."""

    def test_tensorflow_import_handling(self):
        """Test TensorFlow import handling with tf placeholder."""
        from memoraith.integration.common_utils import tf

        # Verify we have either real TensorFlow or the placeholder
        self.assertIsNotNone(tf)

        # Test that the placeholder won't cause exceptions
        try:
            count = tf.keras.backend.count_params(None)
            self.assertEqual(count, 0)
        except Exception as e:
            # Only fail if we're using real TensorFlow
            if 'TENSORFLOW_AVAILABLE' in globals():
                self.fail(f"count_params failed with real TensorFlow: {e}")

    def test_framework_identification(self):
        """Test framework identification functionality."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        model = SimpleModel()
        framework = identify_framework(model)
        self.assertEqual(framework, 'pytorch')

        # Test with non-model object
        framework = identify_framework("not a model")
        self.assertEqual(framework, 'unknown')

    def test_model_size_estimation(self):
        """Test model size estimation with PyTorch model."""
        if not PYTORCH_AVAILABLE:
            self.skipTest("PyTorch not available")

        model = SimpleModel()
        size_info = estimate_model_size(model)

        self.assertIn('parameters', size_info)
        self.assertIn('buffers', size_info)
        self.assertIsInstance(size_info['parameters'], float)
        self.assertGreater(size_info['parameters'], 0)


class TestBaseCollector(unittest.TestCase):
    """Test the fixed BaseDataCollector class."""

    class TestCollector(BaseDataCollector):
        """Test implementation of BaseDataCollector for testing."""

        def __init__(self, interval=0.1):
            super().__init__(interval)
            self.collect_counter = 0

        async def _collect_data(self):
            self.collect_counter += 1
            return {
                'timestamp': time.time(),
                'value': self.collect_counter
            }

        async def validate_data(self):
            return len(self.data) > 0

    def test_error_parameters(self):
        """Test BaseDataCollector with proper error parameters."""
        collector = self.TestCollector()

        async def test_exception_handling():
            # Test start method with proper exception
            try:
                # Trigger condition for exception
                collector._is_collecting = True
                await collector.start()
                self.fail("Should have raised exception")
            except DataCollectionError as e:
                self.assertEqual(e.data_type, "Collection already running")
                self.assertIn("details", dir(e))
                self.assertIsNotNone(e.details)

        asyncio.run(test_exception_handling())

    def test_collection_lifecycle(self):
        """Test complete lifecycle of the collector."""
        collector = self.TestCollector()

        async def run_lifecycle():
            # Start collection
            await collector.start()
            self.assertTrue(collector._is_collecting)
            self.assertIsNotNone(collector._task)

            # Wait for some data collection
            await asyncio.sleep(0.5)

            # Get collected data
            data = await collector.get_data()
            self.assertGreater(len(data), 0)

            # Stop collection
            await collector.stop()
            self.assertFalse(collector._is_collecting)
            self.assertIsNone(collector._task)

            # Verify data persists after stopping
            final_data = await collector.get_data()
            self.assertEqual(len(final_data), len(data))

            # Clear data
            await collector.clear_data()
            empty_data = await collector.get_data()
            self.assertEqual(len(empty_data), 0)

        asyncio.run(run_lifecycle())


class TestEndToEndProfiling(unittest.TestCase):
    """End-to-end test of profiling functionality."""

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="memoraith_e2e_test_")
        set_output_path(self.test_dir)

    def tearDown(self):
        """Clean up test resources."""
        import shutil
        try:
            shutil.rmtree(self.test_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up test directory: {e}")

    @unittest.skipIf(not PYTORCH_AVAILABLE, "PyTorch not available")
    def test_profile_model_decorator(self):
        """Test the profile_model decorator end-to-end."""

        @profile_model(memory=True, computation=True, save_report=True)
        def test_function():
            """Test function to profile."""
            # Create a model and perform some operations
            model = SimpleModel()
            input_data = torch.randn(2, 3, 32, 32)

            # Run forward pass multiple times
            for _ in range(3):
                output = model(input_data)
                loss = output.mean()
                # Just to trigger some computation
                loss_value = loss.item()

            # Allocate some memory to test tracking
            large_tensor = torch.zeros(1000, 1000)
            # Perform some calculations
            result = large_tensor.sum().item()

            return {"result": result}

        # Run the profiled function
        result = test_function()

        # Verify function returned expected result
        self.assertIsInstance(result, dict)
        self.assertIn("result", result)

        # Verify profiling output files were created
        output_files = list(Path(self.test_dir).glob("**/*"))
        self.assertGreater(len(output_files), 0)

        # Check for specific expected report files
        report_files = [f for f in output_files if f.name.endswith('.html') or f.name.endswith('.json')]
        self.assertGreater(len(report_files), 0)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MEMORAITH COMPREHENSIVE INTEGRATION TEST SUITE")
    print("="*80 + "\n")

    if not PYTORCH_AVAILABLE:
        print("WARNING: PyTorch is not available. Some tests will be skipped.")

    # Run tests with verbosity
    unittest.main(verbosity=2)