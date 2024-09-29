import functools
import logging
import asyncio
from typing import Callable, Any, Optional
from .config import config
from .logging_config import setup_logging
from .integration import get_framework_adapter
from .analysis import Analyzer
from .reporting import ReportGenerator
from .exceptions import MemoraithError
from .visualization.real_time_visualizer import RealTimeVisualizer

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
            setup_logging(config.log_level)
            logger = logging.getLogger('memoraith')
            logger.info("Starting Memoraith Profiler...")

            config.enable_memory = memory
            config.enable_time = computation
            config.enable_gpu = gpu

            try:
                model = kwargs.get('model') or args[0]
                adapter = get_framework_adapter(model)

                visualizer = RealTimeVisualizer() if real_time_viz else None

                async with adapter:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(func, *args, **kwargs)

                    if visualizer:
                        await visualizer.update(adapter.data)

                analysis_results = await Analyzer(adapter.data).run_analysis()

                if save_report:
                    await ReportGenerator(analysis_results).generate(format=report_format)

                logger.info("Memoraith Profiling Completed.")
                return result

            except MemoraithError as e:
                logger.error(f"MemoraithError: {e}")
                raise
            except Exception as e:
                logger.exception("An unexpected error occurred during profiling.")
                raise

        return wrapper
    return decorator

def set_output_path(path: str) -> None:
    """Set the output path for profiling reports."""
    config.set_output_path(path)