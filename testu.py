import unittest
import asyncio
import tempfile
import os
import torch
import torch.nn as nn
import tensorflow as tf
from unittest.mock import patch, MagicMock
import pytest
import logging
import time
from memoraith import profile_model, set_output_path
from memoraith.config import Config
from memoraith.exceptions import MemoraithError, FrameworkNotSupportedError
from memoraith.logging_config import setup_logging, get_logger
from memoraith.analysis.analyzer import Analyzer
from memoraith.analysis.anomaly_detection import AnomalyDetector
from memoraith.analysis.bottleneck import BottleneckDetector
from memoraith.analysis.metrics import MetricsCalculator
from memoraith.analysis.recommendations import RecommendationEngine
from memoraith.data_collection.cpu_memory import CPUMemoryTracker
from memoraith.data_collection.gpu_memory import GPUMemoryTracker, PYNVML_AVAILABLE
from memoraith.data_collection.time_tracking import TimeTracker
from memoraith.data_collection.resource_lock import ResourceLock
from memoraith.integration.common_utils import identify_framework, get_model_structure, estimate_model_size
from memoraith.integration.framework_adapter import FrameworkAdapter
from memoraith.integration.pytorch_adapter import PyTorchAdapter
from memoraith.integration.tensorflow_adapter import TensorFlowAdapter
from memoraith.reporting.console_report import ConsoleReport
from memoraith.reporting.export_utils import save_report_as_pdf, export_metrics_to_csv, export_analysis_to_json
from memoraith.reporting.report_generator import ReportGenerator
from memoraith.visualization.heatmap import generate_heatmap
from memoraith.visualization.interactive_dashboard import InteractiveDashboard
from memoraith.visualization.plot_memory import plot_memory_usage
from memoraith.visualization.plot_time import plot_time_usage
from memoraith.visualization.real_time_visualizer import RealTimeVisualizer

@pytest.mark.asyncio
class UnifiedMemoraithTests(unittest.TestCase):
    async def asyncSetUp(self):
        set_output_path('test_profiling_results/')
        self.config = Config()

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        self.model = SimpleModel()

        self.sample_data = {
            'layer1': {'total_time': 0.1, 'total_cpu_memory': 100, 'total_gpu_memory': 200},
            'layer2': {'total_time': 0.2, 'total_cpu_memory': 150, 'total_gpu_memory': 250},
        }
        self.analyzer = Analyzer(self.sample_data)

        self.analysis_results = {
            'metrics': {'global': {'total_time': 1.0, 'peak_cpu_memory': 100, 'peak_gpu_memory': 200}},
            'bottlenecks': [{'layer': 'layer1', 'type': 'time', 'value': 0.5, 'ratio': 0.5}],
            'anomalies': [{'layer': 'layer2', 'type': 'memory', 'value': 1000, 'z_score': 3.0}],
            'recommendations': [{'layer': 'layer3', 'recommendation': 'Optimize this layer'}]
        }

    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.asyncSetUp())

    async def test_setup_logging(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            setup_logging(logging.DEBUG, temp_file.name)
            logger = get_logger("test")
            logger.debug("Test log message")
            temp_file.close()

        await asyncio.sleep(0.1)

        try:
            with open(temp_file.name, 'r') as f:
                content = f.read()
                self.assertIn("Test log message", content)
        finally:
            try:
                os.unlink(temp_file.name)
            except PermissionError:
                pass

    async def test_load_from_env(self):
        with patch.dict('os.environ', {'MEMORAITH_ENABLE_GPU': 'true', 'MEMORAITH_LOG_LEVEL': 'DEBUG'}):
            self.config.load_from_env()
            self.assertTrue(self.config.enable_gpu)
            self.assertEqual(self.config.log_level, 'DEBUG')

    async def test_load_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("enable_gpu: true\nlog_level: DEBUG")
            temp_file.flush()
            self.config.load_from_file(temp_file.name)
        self.assertTrue(self.config.enable_gpu)
        self.assertEqual(self.config.log_level, 'DEBUG')
        os.unlink(temp_file.name)

    async def test_get_optimizer(self):
        mock_params = [torch.nn.Parameter(torch.randn(10, 10))]
        optimizer = self.config.get_optimizer(mock_params)
        self.assertIsNotNone(optimizer)
        self.assertIsInstance(optimizer, torch.optim.Optimizer)

    async def test_get_loss_function(self):
        loss_function = self.config.get_loss_function()
        self.assertIsNotNone(loss_function)

    async def test_memoraith_error(self):
        with self.assertRaises(MemoraithError):
            raise MemoraithError("Test error")

    async def test_framework_not_supported_error(self):
        with self.assertRaises(FrameworkNotSupportedError):
            raise FrameworkNotSupportedError("Test framework")

    async def test_run_analysis(self):
        results = await self.analyzer.run_analysis()
        self.assertIn('metrics', results)
        self.assertIn('bottlenecks', results)
        self.assertIn('anomalies', results)
        self.assertIn('recommendations', results)

    async def test_get_layer_analysis(self):
        layer_analysis = await self.analyzer.get_layer_analysis('layer1')
        self.assertIn('metrics', layer_analysis)
        self.assertIn('bottlenecks', layer_analysis)
        self.assertIn('anomalies', layer_analysis)
        self.assertIn('recommendations', layer_analysis)

    async def test_anomaly_detect(self):
        detector = AnomalyDetector()
        metrics = {
            'layer1': {'total_cpu_memory': 100, 'total_gpu_memory': 200, 'total_time': 0.1},
            'layer2': {'total_cpu_memory': 1000, 'total_gpu_memory': 2000, 'total_time': 1.0},
        }
        anomalies = await detector.detect(metrics)
        self.assertGreater(len(anomalies), 0)

    async def test_bottleneck_detect(self):
        detector = BottleneckDetector()
        metrics = {
            'layer1': {'total_time': 0.1, 'total_cpu_memory': 100},
            'layer2': {'total_time': 1.0, 'total_cpu_memory': 1000},
        }
        bottlenecks = await detector.detect(metrics)
        self.assertGreater(len(bottlenecks), 0)

    async def test_metrics_calculate(self):
        data = {
            'layer1': {'time': 0.1, 'cpu_memory': 100, 'gpu_memory': 200, 'parameters': 1000},
            'layer2': {'time': 0.2, 'cpu_memory': 150, 'gpu_memory': 250, 'parameters': 2000},
        }
        calculator = MetricsCalculator(data)
        metrics = await calculator.calculate()
        self.assertIn('layer1', metrics)
        self.assertIn('layer2', metrics)
        self.assertIn('global', metrics)

    async def test_recommendation_generate(self):
        engine = RecommendationEngine()
        metrics = {
            'layer1': {'total_time': 0.1, 'total_cpu_memory': 100, 'parameters': 1000},
            'layer2': {'total_time': 1.0, 'total_cpu_memory': 1000, 'parameters': 1000000},
        }
        recommendations = await engine.generate(metrics)
        self.assertGreater(len(recommendations), 0)

    async def test_cpu_memory_tracker(self):
        tracker = CPUMemoryTracker()
        tracker.start()
        await asyncio.sleep(0.1)
        tracker.stop()
        peak_memory = tracker.get_peak_memory()
        self.assertIsInstance(peak_memory, dict)

    @unittest.skipIf(not PYNVML_AVAILABLE, "PYNVML not available")
    async def test_gpu_memory_tracker(self):
        tracker = GPUMemoryTracker()
        await tracker.start()
        await asyncio.sleep(0.1)
        await tracker.stop()
        peak_memory = await tracker.get_peak_memory()
        self.assertIsInstance(peak_memory, float)

    async def test_time_tracker(self):
        tracker = TimeTracker()
        tracker.start('test_op')
        await asyncio.sleep(0.1)
        tracker.stop('test_op')
        duration = tracker.get_duration('test_op')
        self.assertIsInstance(duration, float)

    async def test_resource_lock(self):
        lock = ResourceLock('test_lock')
        acquired = lock.acquire()
        self.assertTrue(acquired)
        lock.release()

    async def test_identify_framework(self):
        model = nn.Linear(10, 5)
        framework = identify_framework(model)
        self.assertEqual(framework, 'pytorch')

    async def test_get_model_structure(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        structure = get_model_structure(model)
        self.assertIsInstance(structure, dict)

    async def test_estimate_model_size(self):
        model = nn.Linear(10, 5)
        size = estimate_model_size(model)
        self.assertIn('parameters', size)

    async def test_framework_adapter_abstract_methods(self):
        with self.assertRaises(TypeError):
            FrameworkAdapter()

    async def test_pytorch_adapter(self):
        model = nn.Linear(10, 5)
        adapter = PyTorchAdapter(model)
        await adapter.start_profiling()
        await adapter.stop_profiling()
        input_data = torch.randn(1, 10)
        result = await adapter.profile_inference(input_data)
        self.assertIn('inference_time', result)

    async def test_tensorflow_adapter(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(5, input_shape=(10,))])
        adapter = TensorFlowAdapter(model)
        await adapter.start_profiling()
        await adapter.stop_profiling()
        input_data = tf.random.normal((1, 10))
        result = await adapter.profile_inference(input_data)
        self.assertIn('inference_time', result)

    async def test_console_report(self):
        report = ConsoleReport(self.analysis_results)
        with patch('builtins.print') as mock_print:
            report.display()
            mock_print.assert_called()

    async def test_save_report_as_pdf(self):
        with patch('pdfkit.from_file') as mock_from_file:
            save_report_as_pdf('input.html', 'output.pdf')
            mock_from_file.assert_called_once()

    async def test_export_metrics_to_csv(self):
        metrics = {'layer1': {'total_time': 0.1, 'total_cpu_memory': 100}}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            export_metrics_to_csv(metrics, temp_file.name)
            with open(temp_file.name, 'r') as f:
                content = f.read()
                self.assertIn('layer1', content)
        os.unlink(temp_file.name)

    async def test_export_analysis_to_json(self):
        analysis = {'metrics': {'layer1': {'total_time': 0.1}}}
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            export_analysis_to_json(analysis, temp_file.name)
            with open(temp_file.name, 'r') as f:
                content = f.read()
                self.assertIn('metrics', content)
        os.unlink(temp_file.name)

    async def test_report_generator(self):
        with patch('memoraith.reporting.report_generator.plot_memory_usage') as mock_memory, \
                patch('memoraith.reporting.report_generator.plot_time_usage') as mock_time, \
                patch('memoraith.reporting.report_generator.generate_heatmap') as mock_heatmap, \
                patch('memoraith.reporting.report_generator.InteractiveDashboard') as mock_dashboard:

            generator = ReportGenerator(self.analysis_results)
            with tempfile.TemporaryDirectory() as tmpdir:
                generator.output_path = tmpdir
                await generator.generate()
                mock_memory.assert_called_once()
                mock_time.assert_called_once()
                mock_heatmap.assert_called_once()
                mock_dashboard.return_value.generate.assert_called_once()

    async def test_generate_heatmap(self):
        metrics = {
            'layer1': {'total_cpu_memory': 100, 'total_gpu_memory': 200, 'total_time': 0.1},
            'layer2': {'total_cpu_memory': 150, 'total_gpu_memory': 250, 'total_time': 0.2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_heatmap(metrics, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'metrics_heatmap.png')))

    async def test_interactive_dashboard(self):
        metrics = {
            'layer1': {'total_cpu_memory': 100, 'total_gpu_memory': 200, 'total_time': 0.1, 'parameters': 1000},
            'layer2': {'total_cpu_memory': 150, 'total_gpu_memory': 250, 'total_time': 0.2, 'parameters': 2000},
        }
        dashboard = InteractiveDashboard(metrics)
        with tempfile.TemporaryDirectory() as tmpdir:
            dashboard.generate(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'interactive_dashboard.html')))

    async def test_plot_memory_usage(self):
        metrics = {
            'layer1': {'total_cpu_memory': 100, 'total_gpu_memory': 200},
            'layer2': {'total_cpu_memory': 150, 'total_gpu_memory': 250},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_memory_usage(metrics, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'memory_usage.png')))

    async def test_plot_time_usage(self):
        metrics = {
            'layer1': {'total_time': 0.1},
            'layer2': {'total_time': 0.2},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_time_usage(metrics, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, 'time_usage.png')))

    async def test_real_time_visualizer(self):
        visualizer = RealTimeVisualizer()
        data = {
            'layer1': {'cpu_memory': 100, 'time': 0.1},
            'layer2': {'cpu_memory': 150, 'time': 0.2},
        }
        with patch('matplotlib.pyplot.show') as mock_show:
            await visualizer.update(data)
            mock_show.assert_called()

    async def test_real_time_visualizer_close(self):
        visualizer = RealTimeVisualizer()
        with patch('matplotlib.pyplot.close') as mock_close:
            visualizer.close()
            mock_close.assert_called_once()

    @profile_model()
    async def dummy_train(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(10):
            input_data = torch.randn(32, 10)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()

    async def test_profiler_decorator(self):
        try:
            await self.dummy_train(self.model)
        except Exception as e:
            self.fail(f"Profiler decorator raised an exception: {e}")

    async def test_output_files(self):
        await self.dummy_train(self.model)
        self.assertTrue(os.path.exists('test_profiling_results/memoraith_report.html'))
        self.assertTrue(os.path.exists('test_profiling_results/memory_usage.png'))
        self.assertTrue(os.path.exists('test_profiling_results/time_usage.png'))
        self.assertTrue(os.path.exists('test_profiling_results/metrics_heatmap.png'))
        self.assertTrue(os.path.exists('test_profiling_results/interactive_dashboard.html'))

    async def test_config_loading(self):
        config = Config()
        config.load_from_file('test_config.yaml')
        self.assertEqual(config.enable_gpu, True)
        self.assertEqual(config.log_level, 'DEBUG')

if __name__ == '__main__':
    pytest.main([__file__])