import os
import asyncio
from typing import Dict, Any
import aiofiles
import logging
from jinja2 import Environment, FileSystemLoader
from ..visualization import plot_memory_usage, plot_time_usage, generate_heatmap, InteractiveDashboard
from ..config import config
from .export_utils import save_report_as_pdf, export_metrics_to_csv, export_analysis_to_json, create_export_directory

class ReportGenerator:
    """Generates comprehensive reports from the analysis results with enhanced functionality."""

    def __init__(self, analysis_results: Dict[str, Any]):
        self.analysis_results = analysis_results
        self.output_path = str(config.output_path)
        self.logger = logging.getLogger(__name__)
        self.export_dir = create_export_directory(self.output_path)

    async def generate(self, format: str = 'html') -> None:
        """
        Generate the report and save it to the output path.

        Args:
            format (str): The format of the report ('html' or 'pdf')
        """
        try:
            # Generate visualizations asynchronously
            await asyncio.gather(
                plot_memory_usage(self.analysis_results['metrics'], self.export_dir),
                plot_time_usage(self.analysis_results['metrics'], self.export_dir),
                generate_heatmap(self.analysis_results['metrics'], self.export_dir),
                InteractiveDashboard(self.analysis_results['metrics']).generate(self.export_dir)
            )

            # Generate HTML report
            html_content = self._generate_html_report()
            html_path = os.path.join(self.export_dir, 'memoraith_report.html')
            await self._save_html_report(html_content, html_path)

            # Generate PDF if requested
            if format.lower() == 'pdf':
                pdf_path = os.path.join(self.export_dir, 'memoraith_report.pdf')
                await self._save_pdf_report(html_path, pdf_path)

            # Export additional data
            await asyncio.gather(
                self._export_metrics_csv(),
                self._export_analysis_json()
            )

            self.logger.info(f"Report generation completed. Files saved in {self.export_dir}")
        except Exception as e:
            self.logger.error(f"Error during report generation: {str(e)}")
            raise

    def _generate_html_report(self) -> str:
        """Generate the HTML content for the report."""
        try:
            template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
            env = Environment(loader=FileSystemLoader(template_dir))
            template = env.get_template('report_template.html')
            return template.render(
                analysis_results=self.analysis_results,
                config=config,
                memory_plot_path='memory_usage.png',
                time_plot_path='time_usage.png',
                heatmap_path='metrics_heatmap.png',
                dashboard_path='interactive_dashboard.html'
            )
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            raise

    async def _save_html_report(self, content: str, file_path: str) -> None:
        """Save the HTML report to a file."""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            self.logger.info(f"HTML report saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving HTML report: {str(e)}")
            raise

    async def _save_pdf_report(self, html_path: str, pdf_path: str) -> None:
        """Save the report as a PDF file."""
        try:
            await asyncio.to_thread(save_report_as_pdf, html_path, pdf_path)
        except Exception as e:
            self.logger.error(f"Error saving PDF report: {str(e)}")
            raise

    async def _export_metrics_csv(self) -> None:
        """Export metrics to a CSV file."""
        csv_path = os.path.join(self.export_dir, 'metrics.csv')
        await asyncio.to_thread(export_metrics_to_csv, self.analysis_results['metrics'], csv_path)

    async def _export_analysis_json(self) -> None:
        """Export the full analysis results to a JSON file."""
        json_path = os.path.join(self.export_dir, 'analysis_results.json')
        await asyncio.to_thread(export_analysis_to_json, self.analysis_results, json_path)

    def get_report_files(self) -> Dict[str, str]:
        """Get a dictionary of generated report files and their paths."""
        return {
            'html_report': os.path.join(self.export_dir, 'memoraith_report.html'),
            'pdf_report': os.path.join(self.export_dir, 'memoraith_report.pdf'),
            'metrics_csv': os.path.join(self.export_dir, 'metrics.csv'),
            'analysis_json': os.path.join(self.export_dir, 'analysis_results.json'),
            'memory_plot': os.path.join(self.export_dir, 'memory_usage.png'),
            'time_plot': os.path.join(self.export_dir, 'time_usage.png'),
            'heatmap': os.path.join(self.export_dir, 'metrics_heatmap.png'),
            'interactive_dashboard': os.path.join(self.export_dir, 'interactive_dashboard.html')
        }