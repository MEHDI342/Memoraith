import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def plot_memory_usage(metrics: Dict[str, Any], output_path: str, figsize: tuple = (12, 6), colors: tuple = ('skyblue', 'orange')) -> None:
    """
    Generate an enhanced bar chart for memory usage per layer.

    Args:
        metrics (Dict[str, Any]): Calculated metrics for each layer
        output_path (str): Path to save the generated plot
        figsize (tuple): Figure size (width, height) in inches
        colors (tuple): Colors for CPU and GPU memory bars
    """
    try:
        layers = []
        cpu_memory = []
        gpu_memory = []

        for layer, layer_metrics in metrics.items():
            if isinstance(layer_metrics, dict):
                layers.append(layer)
                cpu_memory.append(layer_metrics.get('total_cpu_memory', 0))
                gpu_memory.append(layer_metrics.get('total_gpu_memory', 0))

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(layers))
        bar_width = 0.35

        cpu_bars = ax.bar(x - bar_width/2, cpu_memory, bar_width, label='CPU Memory', color=colors[0])
        gpu_bars = ax.bar(x + bar_width/2, gpu_memory, bar_width, label='GPU Memory', color=colors[1])

        ax.set_xlabel('Layers', fontsize=12)
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Usage per Layer', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.legend()

        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=90)

        autolabel(cpu_bars)
        autolabel(gpu_bars)

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{output_path}/memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Memory usage plot generated and saved to {output_path}/memory_usage.png")
    except Exception as e:
        logger.error(f"Error generating memory usage plot: {str(e)}")
        raise
