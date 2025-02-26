"""
Real-time visualization of profiling data during model execution.
"""

import logging
import matplotlib.pyplot as plt
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

class RealTimeVisualizer:
    """Real-time visualization of profiling data."""

    def __init__(self):
        """Initialize the visualizer with appropriate plots."""
        self.logger = logging.getLogger(__name__)
        self.memory_data = {}
        self.time_data = {}
        self.network_data = {}
        self._fig = None
        self._axes = None
        self._is_running = False
        self._update_task = None

        # Setup the visualization
        self._setup_visualization()

    def _setup_visualization(self) -> None:
        """Setup the visualization plots."""
        try:
            plt.ion()  # Turn on interactive mode
            self._fig, self._axes = plt.subplots(2, 1, figsize=(10, 10))
            self._fig.tight_layout(pad=3.0)
            self._fig.suptitle('Memoraith Real-time Profiling', fontsize=16)

            # Setup memory plot
            self._axes[0].set_title('Memory Usage by Layer')
            self._axes[0].set_xlabel('Memory Usage (MB)')
            self._axes[0].set_ylabel('Layers')

            # Setup time plot
            self._axes[1].set_title('Computation Time by Layer')
            self._axes[1].set_xlabel('Time (s)')
            self._axes[1].set_ylabel('Layers')

            plt.tight_layout()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()

        except Exception as e:
            self.logger.error(f"Error setting up visualization: {str(e)}")

    async def start(self) -> None:
        """Start the real-time visualization update loop."""
        if self._is_running:
            self.logger.warning("Visualization is already running")
            return

        self._is_running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Real-time visualization started")

    async def _update_loop(self) -> None:
        """Real-time visualization update loop."""
        try:
            while self._is_running:
                if self.memory_data and self.time_data:
                    await self._draw()
                await asyncio.sleep(0.5)  # Update every 500ms
        except Exception as e:
            self.logger.error(f"Error in visualization update loop: {str(e)}")
        finally:
            self.logger.info("Visualization update loop stopped")

    async def update(self, data: Dict[str, Any]) -> None:
        """
        Update the visualization with new data.
        """
        try:
            for layer, layer_data in data.items():
                if not isinstance(layer_data, dict):
                    continue

                # Extract memory data
                if 'total_cpu_memory' in layer_data:
                    self.memory_data[layer] = layer_data['total_cpu_memory']
                elif 'cpu_memory' in layer_data:
                    self.memory_data[layer] = layer_data['cpu_memory']

                # Extract time data
                if 'total_time' in layer_data:
                    self.time_data[layer] = layer_data['total_time']
                elif 'time' in layer_data:
                    self.time_data[layer] = layer_data['time']

                # Extract network data if available
                if 'network_bandwidth' in layer_data:
                    self.network_data[layer] = layer_data['network_bandwidth']

            if not self._is_running:
                # Start the update loop if not already running
                await self.start()

        except Exception as e:
            self.logger.error(f"Error updating visualization data: {str(e)}")

    async def _draw(self) -> None:
        """Draw the updated visualization."""
        try:
            # Clear the axes
            for ax in self._axes:
                ax.clear()

            # Get data
            layers = list(set(list(self.memory_data.keys()) + list(self.time_data.keys())))

            if not layers:
                return

            # Prepare memory data
            memory_values = [self.memory_data.get(layer, 0) for layer in layers]

            # Sort by memory usage (descending)
            memory_data = sorted(zip(layers, memory_values), key=lambda x: x[1], reverse=True)
            sorted_layers = [item[0] for item in memory_data]
            sorted_memory = [item[1] for item in memory_data]

            # Draw memory plot
            bars = self._axes[0].barh(sorted_layers, sorted_memory, color='skyblue')
            self._axes[0].set_title('Memory Usage by Layer')
            self._axes[0].set_xlabel('Memory Usage (MB)')

            # Add value labels
            for i, v in enumerate(sorted_memory):
                if v > 0:  # Only add labels for non-zero values
                    self._axes[0].text(v + 1, i, f'{v:.2f} MB', va='center')

            # Prepare time data
            time_values = [self.time_data.get(layer, 0) for layer in sorted_layers]

            # Draw time plot
            bars = self._axes[1].barh(sorted_layers, time_values, color='salmon')
            self._axes[1].set_title('Computation Time by Layer')
            self._axes[1].set_xlabel('Time (s)')

            # Add value labels
            for i, v in enumerate(time_values):
                if v > 0:  # Only add labels for non-zero values
                    self._axes[1].text(v + 0.01, i, f'{v:.4f} s', va='center')

            # Adjust layout and draw
            plt.tight_layout()
            await asyncio.to_thread(self._fig.canvas.draw)
            await asyncio.to_thread(self._fig.canvas.flush_events)

        except Exception as e:
            self.logger.error(f"Error drawing visualization: {str(e)}")

    async def stop(self) -> None:
        """Stop the visualization."""
        self._is_running = False
        if self._update_task:
            try:
                self._update_task.cancel()
                await asyncio.gather(self._update_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            self._update_task = None

        # Close the plot
        await asyncio.to_thread(plt.close, self._fig)
        self.logger.info("Real-time visualization stopped")