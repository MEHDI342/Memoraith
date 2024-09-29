import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def plot_time_usage(metrics: Dict[str, Any], output_path: str, figsize: tuple = (12, 6), color: str = 'salmon') -> None:
    """
    Generate an enhanced bar chart for computation time per layer.

    Args:
        metrics (Dict[str, Any]): Calculated metrics for each layer
        output_path (str): Path to save the generated plot
        figsize (tuple): Figure size (width, height) in inches
        color (str): Color for the time usage bars
    """
    try:
        layers = []
        times = []

        for layer, layer_metrics in metrics.items():
            if isinstance(layer_metrics, dict):
                layers.append(layer)
                times.append(layer_metrics.get('total_time', 0))

        fig, ax = plt.subplots(figsize=figsize)

        bars = ax.barh(layers, times, color=color)

        ax.set_xlabel('Computation Time (s)', fontsize=12)
        ax.set_ylabel('Layers', fontsize=12)
        ax.set_title('Computation Time per Layer', fontsize=16)

        # Add value labels at the end of bars
        for i, v in enumerate(times):
            ax.text(v, i, f' {v:.4f}s', va='center')

        # Highlight the top 3 time-consuming layers
        top_3_indices = np.argsort(times)[-3:]
        for i in top_3_indices:
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)

        # Add a text box with summary statistics
        total_time = sum(times)
        avg_time = np.mean(times)
        textstr = f'Total Time: {total_time:.4f}s\nAvg Time: {avg_time:.4f}s'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{output_path}/time_usage.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Time usage plot generated and saved to {output_path}/time_usage.png")
    except Exception as e:
        logger.error(f"Error generating time usage plot: {str(e)}")
        raise