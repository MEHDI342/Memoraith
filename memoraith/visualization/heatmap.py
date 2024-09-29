import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def generate_heatmap(metrics: Dict[str, Any], output_path: str, figsize: tuple = (12, 8), cmap: str = 'viridis', annot: bool = True) -> None:
    """
    Generate an enhanced heatmap showing memory and time intensity.

    Args:
        metrics (Dict[str, Any]): Calculated metrics for each layer
        output_path (str): Path to save the generated heatmap
        figsize (tuple): Figure size (width, height) in inches
        cmap (str): Colormap for the heatmap
        annot (bool): Whether to annotate each cell with the value
    """
    try:
        data = []
        for layer, layer_metrics in metrics.items():
            if isinstance(layer_metrics, dict):
                data.append({
                    'Layer': layer,
                    'CPU Memory (MB)': layer_metrics.get('total_cpu_memory', 0),
                    'GPU Memory (MB)': layer_metrics.get('total_gpu_memory', 0),
                    'Time (s)': layer_metrics.get('total_time', 0)
                })

        df = pd.DataFrame(data)
        df.set_index('Layer', inplace=True)

        # Normalize data for better visualization
        for column in df.columns:
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

        plt.figure(figsize=figsize)
        ax = sns.heatmap(df, annot=annot, cmap=cmap, fmt='.2f',
                         linewidths=0.5, cbar_kws={'label': 'Normalized Intensity'})

        plt.title('Layer Metrics Heatmap (Normalized)', fontsize=16)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Layers', fontsize=12)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{output_path}/metrics_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Heatmap generated and saved to {output_path}/metrics_heatmap.png")
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")
        raise