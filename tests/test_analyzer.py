import unittest
from memoraith.analysis.analyzer import Analyzer

class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.sample_data = {
            'layer1': {'total_time': 0.1, 'total_cpu_memory': 100, 'total_gpu_memory': 200},
            'layer2': {'total_time': 0.2, 'total_cpu_memory': 150, 'total_gpu_memory': 250},
        }
        self.analyzer = Analyzer(self.sample_data)

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

if __name__ == '__main__':
    unittest.main()