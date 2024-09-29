from typing import Dict, Any
import logging
from colorama import Fore, Style, init

init(autoreset=True)  # Initialize colorama

class ConsoleReport:
    """Generates a comprehensive console report from the analysis results."""

    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.logger = logging.getLogger(__name__)

    def display(self) -> None:
        """Display the report in the console with enhanced formatting and colors."""
        try:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}==== Memoraith Profiling Report ===={Style.RESET_ALL}")

            self._display_global_metrics()
            self._display_top_consumers('time', 'Time Consumers', 'total_time', 's')
            self._display_top_consumers('cpu_memory', 'CPU Memory Consumers', 'total_cpu_memory', 'MB')
            self._display_top_consumers('gpu_memory', 'GPU Memory Consumers', 'total_gpu_memory', 'MB')
            self._display_bottlenecks()
            self._display_recommendations()
            self._display_anomalies()

            print(f"\n{Fore.YELLOW}For detailed visualizations and interactive dashboard, please refer to the generated HTML report.")
        except Exception as e:
            self.logger.error(f"Error displaying console report: {str(e)}")

    def _display_global_metrics(self) -> None:
        """Display global metrics."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Global Metrics:{Style.RESET_ALL}")
        global_metrics = self.analysis_results['metrics'].get('global', {})
        print(f"Total Time: {Fore.YELLOW}{global_metrics.get('total_time', 0):.4f} s{Style.RESET_ALL}")
        print(f"Peak CPU Memory: {Fore.YELLOW}{global_metrics.get('peak_cpu_memory', 0):.2f} MB{Style.RESET_ALL}")
        print(f"Peak GPU Memory: {Fore.YELLOW}{global_metrics.get('peak_gpu_memory', 0):.2f} MB{Style.RESET_ALL}")
        print(f"Total Parameters: {Fore.YELLOW}{global_metrics.get('total_parameters', 0):,}{Style.RESET_ALL}")
        if 'total_flops' in global_metrics:
            print(f"Total FLOPs: {Fore.YELLOW}{global_metrics['total_flops']:,}{Style.RESET_ALL}")

    def _display_top_consumers(self, metric_type: str, title: str, metric_key: str, unit: str, top_n: int = 5) -> None:
        """Display top consumers for a specific metric."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Top {top_n} {title}:{Style.RESET_ALL}")
        sorted_layers = sorted(
            [(layer, metrics[metric_key]) for layer, metrics in self.analysis_results['metrics'].items() if isinstance(metrics, dict)],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

        for layer, value in sorted_layers:
            print(f"Layer: {Fore.CYAN}{layer}{Style.RESET_ALL}, {metric_type.capitalize()}: {Fore.YELLOW}{value:.4f} {unit}{Style.RESET_ALL}")

    def _display_bottlenecks(self) -> None:
        """Display detected bottlenecks."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Detected Bottlenecks:{Style.RESET_ALL}")
        for bottleneck in self.analysis_results['bottlenecks']:
            print(f"Layer: {Fore.CYAN}{bottleneck['layer']}{Style.RESET_ALL}, "
                  f"Type: {Fore.MAGENTA}{bottleneck['type']}{Style.RESET_ALL}, "
                  f"Value: {Fore.YELLOW}{bottleneck['value']:.4f}{Style.RESET_ALL}, "
                  f"Ratio: {Fore.YELLOW}{bottleneck['ratio']:.2%}{Style.RESET_ALL}")

    def _display_recommendations(self) -> None:
        """Display optimization recommendations."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Recommendations:{Style.RESET_ALL}")
        for rec in self.analysis_results['recommendations']:
            print(f"Layer: {Fore.CYAN}{rec['layer']}{Style.RESET_ALL}")
            print(f"Recommendation: {Fore.YELLOW}{rec['recommendation']}{Style.RESET_ALL}")
            print()

    def _display_anomalies(self) -> None:
        """Display detected anomalies."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Detected Anomalies:{Style.RESET_ALL}")
        for anomaly in self.analysis_results['anomalies']:
            print(f"Layer: {Fore.CYAN}{anomaly['layer']}{Style.RESET_ALL}, "
                  f"Type: {Fore.MAGENTA}{anomaly['type']}{Style.RESET_ALL}, "
                  f"Value: {Fore.YELLOW}{anomaly['value']:.4f}{Style.RESET_ALL}, "
                  f"Z-Score: {Fore.YELLOW}{anomaly.get('z_score', 'N/A'):.2f}{Style.RESET_ALL}")

    def _display_performance_score(self) -> None:
        """Display the overall performance score."""
        if 'performance_score' in self.analysis_results:
            score = self.analysis_results['performance_score']
            color = Fore.GREEN if score > 80 else (Fore.YELLOW if score > 60 else Fore.RED)
            print(f"\n{Fore.GREEN}{Style.BRIGHT}Overall Performance Score:{Style.RESET_ALL}")
            print(f"{color}{score:.2f}/100{Style.RESET_ALL}")

    def save_to_file(self, file_path: str) -> None:
        """Save the console report to a text file."""
        try:
            with open(file_path, 'w') as f:
                # Redirect print output to the file
                import sys
                original_stdout = sys.stdout
                sys.stdout = f
                self.display()
                sys.stdout = original_stdout
            self.logger.info(f"Console report saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving console report to file: {str(e)}")