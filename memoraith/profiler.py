import functools
import logging
import asyncio
from typing import Callable, Any, Dict
from pathlib import Path

from memoraith.config import config
from memoraith.logging_config import setup_logging
from memoraith.integration import get_framework_adapter
from memoraith.analysis import Analyzer
from memoraith.reporting import ReportGenerator
from memoraith.exceptions import MemoraithError
from memoraith.visualization.real_time_visualizer import RealTimeVisualizer

def profile_model(
        memory: bool = True,
        computation: bool = True,
        gpu: bool = False,
        save_report: bool = True,
        report_format: str = 'html',
        real_time_viz: bool = False
) -> Callable:
    """
    Decorator to profile a model's training or inference function.

    Args:
        memory (bool): Enable memory profiling
        computation (bool): Enable computation time profiling
        gpu (bool): Enable GPU profiling
        save_report (bool): Save the profiling report
        report_format (str): Format of the saved report ('html' or 'pdf')
        real_time_viz (bool): Enable real-time visualization

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Setup logging
            setup_logging(config.log_level)
            logger = logging.getLogger('memoraith')
            logger.info("Starting Memoraith Profiler...")

            # Update configuration
            config.enable_memory = memory
            config.enable_time = computation
            config.enable_gpu = gpu
            config.report_format = report_format

            visualizer = None
            try:
                # Get model from args or kwargs
                model = kwargs.get('model')
                if model is None and args:
                    model = args[0]

                if not model:
                    raise MemoraithError("No model provided for profiling")

                # Initialize framework adapter
                adapter = get_framework_adapter(model)

                # Initialize visualizer if requested
                if real_time_viz:
                    visualizer = RealTimeVisualizer()

                # Profile the model
                async with adapter:
                    # Execute the wrapped function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)

                    # Update visualization if enabled
                    if visualizer and adapter.data:
                        await visualizer.update(adapter.data)

                    # Analyze results
                    analyzer = Analyzer(adapter.data)
                    analysis_results = await analyzer.run_analysis()

                    # Generate report if requested
                    if save_report:
                        report_generator = ReportGenerator(analysis_results)
                        await report_generator.generate(format=report_format)

                    logger.info("Memoraith Profiling Completed Successfully")
                    return result

            except MemoraithError as e:
                logger.error(f"MemoraithError: {str(e)}")
                raise
            except Exception as e:
                logger.exception("An unexpected error occurred during profiling")
                raise
            finally:
                if visualizer:
                    try:
                        await visualizer.stop()
                    except Exception as e:
                        logger.error(f"Error stopping visualizer: {str(e)}")

        return wrapper
    return decorator

def set_output_path(path: str) -> None:
    """
    Set the output path for profiling reports.

    Args:
        path (str): Directory path for saving profiling outputs
    """
    try:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        config.output_path = output_path
        logging.getLogger('memoraith').info(f"Output path set to: {output_path}")
    except Exception as e:
        raise MemoraithError(f"Failed to set output path: {str(e)}")