import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any
import logging

class InteractiveDashboard:
    """
    Generates a comprehensive interactive dashboard for exploring profiling data.
    """

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics
        self.logger = logging.getLogger(__name__)

    def generate(self, output_path: str) -> None:
        """
        Generate an advanced interactive dashboard and save it as an HTML file.

        Args:
            output_path (str): Path to save the generated dashboard
        """
        try:
            df = pd.DataFrame.from_dict(self.metrics, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Layer'}, inplace=True)

            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Memory Usage", "Computation Time",
                    "Parameters Count", "Layer Efficiency",
                    "Memory vs Time", "Cumulative Metrics"
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )

            # Memory Usage
            fig.add_trace(go.Bar(x=df['Layer'], y=df['total_cpu_memory'], name='CPU Memory'),
                          row=1, col=1)
            fig.add_trace(go.Bar(x=df['Layer'], y=df['total_gpu_memory'], name='GPU Memory'),
                          row=1, col=1)

            # Computation Time
            fig.add_trace(go.Bar(x=df['Layer'], y=df['total_time'], name='Time'),
                          row=1, col=2)

            # Parameters Count
            fig.add_trace(go.Bar(x=df['Layer'], y=df['parameters'], name='Parameters'),
                          row=2, col=1)

            # Layer Efficiency
            efficiency = df['parameters'] / df['total_time']
            fig.add_trace(go.Scatter(x=df['Layer'], y=efficiency, mode='lines+markers', name='Efficiency'),
                          row=2, col=2)

            # Memory vs Time Scatter
            fig.add_trace(go.Scatter(x=df['total_time'], y=df['total_cpu_memory'], mode='markers', name='CPU Memory vs Time'),
                          row=3, col=1)
            fig.add_trace(go.Scatter(x=df['total_time'], y=df['total_gpu_memory'], mode='markers', name='GPU Memory vs Time'),
                          row=3, col=1)

            # Cumulative Metrics
            df_sorted = df.sort_values('total_time')
            df_sorted['cumulative_time'] = df_sorted['total_time'].cumsum()
            df_sorted['cumulative_memory'] = df_sorted['total_cpu_memory'].cumsum()
            fig.add_trace(go.Scatter(x=df_sorted['Layer'], y=df_sorted['cumulative_time'], mode='lines', name='Cumulative Time'),
                          row=3, col=2)
            fig.add_trace(go.Scatter(x=df_sorted['Layer'], y=df_sorted['cumulative_memory'], mode='lines', name='Cumulative Memory'),
                          row=3, col=2)

            fig.update_layout(height=1200, width=1600, title_text="Memoraith Advanced Profiling Results")
            fig.write_html(f"{output_path}/interactive_dashboard.html")
            self.logger.info(f"Interactive dashboard generated and saved to {output_path}/interactive_dashboard.html")
        except Exception as e:
            self.logger.error(f"Error generating interactive dashboard: {str(e)}")

    def generate_layer_comparison(self, output_path: str) -> None:
        """
        Generate a separate dashboard for layer-by-layer comparison.

        Args:
            output_path (str): Path to save the generated dashboard
        """
        try:
            df = pd.DataFrame.from_dict(self.metrics, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Layer'}, inplace=True)

            fig = go.Figure()

            for metric in ['total_cpu_memory', 'total_gpu_memory', 'total_time', 'parameters']:
                fig.add_trace(go.Bar(x=df['Layer'], y=df[metric], name=metric))

            fig.update_layout(
                barmode='group',
                height=600,
                width=1200,
                title_text="Layer-by-Layer Comparison",
                xaxis_title="Layers",
                yaxis_title="Metric Value (log scale)",
                yaxis_type="log"
            )

            fig.write_html(f"{output_path}/layer_comparison.html")
            self.logger.info(f"Layer comparison dashboard generated and saved to {output_path}/layer_comparison.html")
        except Exception as e:
            self.logger.error(f"Error generating layer comparison dashboard: {str(e)}")