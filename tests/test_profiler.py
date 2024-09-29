import unittest
import asyncio
import torch
import torch.nn as nn
import tensorflow as tf
from memoraith import profile_model, set_output_path
from memoraith.config import Config
from memoraith.data_collection.gpu_memory import PYNVML_AVAILABLE

class TestProfiler(unittest.TestCase):
    def setUp(self):
        set_output_path('test_profiling_results/')

        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        self.model = SimpleModel()

    @profile_model()
    async def dummy_train(self, model):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(10):
            input_data = torch.randn(32, 10)
            output = model(input_data)
            loss = output.sum()
            loss.backward()
            optimizer.step()

    @unittest.skipIf(not PYNVML_AVAILABLE, "PYNVML not available")
    async def test_profiler_decorator(self):
        try:
            await self.dummy_train(self.model)
        except Exception as e:
            self.fail(f"Profiler decorator raised an exception: {e}")

    async def test_output_files(self):
        import os
        await self.dummy_train(self.model)
        self.assertTrue(os.path.exists('test_profiling_results/memoraith_report.html'))
        self.assertTrue(os.path.exists('test_profiling_results/memory_usage.png'))
        self.assertTrue(os.path.exists('test_profiling_results/time_usage.png'))
        self.assertTrue(os.path.exists('test_profiling_results/metrics_heatmap.png'))
        self.assertTrue(os.path.exists('test_profiling_results/interactive_dashboard.html'))

    def test_config_loading(self):
        config = Config()
        config.load_from_file('test_config.yaml')
        self.assertEqual(config.enable_gpu, True)
        self.assertEqual(config.log_level, 'DEBUG')

if __name__ == '__main__':
    unittest.main()
