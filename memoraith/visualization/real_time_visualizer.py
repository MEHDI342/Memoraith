import asyncio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, Any
import numpy as np

class RealTimeVisualizer:
    """Real-time visualization of profiling data."""

    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        self.memory_data = {}
        self.time_data = {}
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

    async def add_data(self, memory, time):
        self.memory_data.append(memory)
        self.time_data.append(time)

    async def stop(self):
        if self.animation:
            self.animation.event_source.stop()
        plt.close(self.fig)

# Usage
async def main():
    visualizer = RealTimeVisualizer()
    await visualizer.start()

    for _ in range(100):
        await visualizer.add_data(np.random.rand() * 100, np.random.rand())
        await asyncio.sleep(0.1)

    await visualizer.stop()

if __name__ == "__main__":
    asyncio.run(main())

    # Usage
    async def main():
        visualizer = RealTimeVisualizer()
        await visualizer.start()

        for _ in range(100):
            await visualizer.add_data(np.random.rand() * 100, np.random.rand())
            await asyncio.sleep(0.1)

        await visualizer.stop()

    if __name__ == "__main__":
        asyncio.run(main())