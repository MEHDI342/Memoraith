import logging
import numpy as np
import matplotlib.pyplot as plt
import aiofiles
from typing import Dict, Any, List
import asyncio
from ..config import config

# 2. Fix real_time_visualizer.py
class RealTimeVisualizer:
    """Real-time visualization of profiling data."""

    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        self.memory_data = {}
        self.time_data = {}
        self.animation = None
        plt.ion()  # Turn on interactive mode
        self.fig.show()

    async def update(self, data: Dict[str, Any]) -> None:
        """Update the visualization with new data."""
        for layer, layer_data in data.items():
            self.memory_data[layer] = layer_data.get('cpu_memory', 0)
            self.time_data[layer] = layer_data.get('time', 0)

        await self._draw()

    async def _draw(self) -> None:
        """Draw the updated visualization."""
        self.ax1.clear()
        self.ax2.clear()

        layers = list(self.memory_data.keys())
        memory_values = list(self.memory_data.values())
        time_values = list(self.time_data.values())

        self.ax1.barh(layers, memory_values)
        self.ax1.set_xlabel('Memory Usage (MB)')
        self.ax1.set_title('Real-time Memory Usage by Layer')

        self.ax2.barh(layers, time_values)
        self.ax2.set_xlabel('Computation Time (s)')
        self.ax2.set_title('Real-time Computation Time by Layer')

        plt.tight_layout()
        await asyncio.to_thread(self.fig.canvas.draw)
        await asyncio.to_thread(self.fig.canvas.flush_events)

    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)

    async def add_data(self, memory: float, time: float) -> None:
        """Add new data points to the visualization."""
        self.memory_data = memory
        self.time_data = time
        await self._draw()

    async def stop(self) -> None:
        """Stop the visualization."""
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)

# 3. Fix TensorFlowAdapter _wrapped_call method
async def _wrapped_call(self, *args, **kwargs) -> Any:
    """Wrapped call method for profiling each layer."""
    output = None
    for layer in self.model.layers:
        layer_name = f"{layer.__class__.__name__}_{id(layer)}"
        self.time_tracker.start(layer_name)
        output = layer(*args, **kwargs)
        self.time_tracker.stop(layer_name)

        try:
            self.data[layer_name] = self.data.get(layer_name, {})
            self.data[layer_name]['time'] = self.time_tracker.get_duration(layer_name)
            self.data[layer_name]['parameters'] = layer.count_params()

            if config.enable_memory:
                self.data[layer_name]['cpu_memory'] = await self.cpu_tracker.get_peak_memory()
                if self.gpu_tracker:
                    self.data[layer_name]['gpu_memory'] = await self.gpu_tracker.get_peak_memory()

            if hasattr(output, 'shape'):
                self.data[layer_name]['output_shape'] = output.shape.as_list()
        except Exception as e:
            self.logger.error(f"Error in _wrapped_call for layer {layer_name}: {str(e)}")

        args = (output,)

    return output