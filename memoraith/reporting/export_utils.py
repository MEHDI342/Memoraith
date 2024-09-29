from typing import Optional, Dict, Any
import pdfkit
import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def save_report_as_pdf(html_report_path: str, pdf_output_path: str, config: Optional[Dict[str, Any]] = None) -> None:
    """
    Convert HTML report to PDF with enhanced error handling and logging.

    Args:
        html_report_path (str): Path to the HTML report file
        pdf_output_path (str): Path where the PDF report should be saved
        config (Optional[Dict[str, Any]]): Configuration options for pdfkit
    """
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None
    }

    if config:
        options.update(config)

    try:
        pdfkit.from_file(html_report_path, pdf_output_path, options=options)
        logger.info(f"PDF report saved successfully at: {pdf_output_path}")
    except OSError as e:
        if 'wkhtmltopdf' in str(e):
            logger.error("wkhtmltopdf is not installed or not found in the system PATH.")
            logger.info("Please install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        else:
            logger.error(f"OS error occurred while generating PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while generating PDF: {str(e)}")

def export_metrics_to_csv(metrics: Dict[str, Any], csv_output_path: str) -> None:
    """
    Export metrics data to a CSV file.

    Args:
        metrics (Dict[str, Any]): Metrics data to be exported
        csv_output_path (str): Path where the CSV file should be saved
    """
    try:
        with open(csv_output_path, 'w', newline='') as csvfile:
            fieldnames = ['layer', 'total_time', 'total_cpu_memory', 'total_gpu_memory', 'parameters']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for layer, data in metrics.items():
                if isinstance(data, dict):
                    writer.writerow({
                        'layer': layer,
                        'total_time': data.get('total_time', 'N/A'),
                        'total_cpu_memory': data.get('total_cpu_memory', 'N/A'),
                        'total_gpu_memory': data.get('total_gpu_memory', 'N/A'),
                        'parameters': data.get('parameters', 'N/A')
                    })
        logger.info(f"Metrics exported to CSV: {csv_output_path}")
    except Exception as e:
        logger.error(f"Error exporting metrics to CSV: {str(e)}")

def export_analysis_to_json(analysis_results: Dict[str, Any], json_output_path: str) -> None:
    """
    Export the full analysis results to a JSON file.

    Args:
        analysis_results (Dict[str, Any]): Analysis results to be exported
        json_output_path (str): Path where the JSON file should be saved
    """
    try:
        with open(json_output_path, 'w') as jsonfile:
            json.dump(analysis_results, jsonfile, indent=2)
        logger.info(f"Analysis results exported to JSON: {json_output_path}")
    except Exception as e:
        logger.error(f"Error exporting analysis results to JSON: {str(e)}")

def create_export_directory(base_path: str) -> str:
    """
    Create a directory for exporting files if it doesn't exist.

    Args:
        base_path (str): Base path for creating the export directory

    Returns:
        str: Path to the created export directory
    """
    export_dir = Path(base_path) / "memoraith_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Export directory created: {export_dir}")
    return str(export_dir)