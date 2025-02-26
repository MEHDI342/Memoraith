"""
Memoraith Core Profiler - Main entry point for profiling functionality.
"""

import functools
import logging
import asyncio
import inspect
from typing import Callable, Any, Dict, Optional, TypeVar, cast
from pathlib import Path

from memoraith.config import config
from memoraith.logging_config import setup_logging
from memoraith.integration import get_framework_adapter
from memoraith.analysis import Analyzer
from memoraith.reporting import ReportGenerator
from memoraith.exceptions import MemoraithError
from memoraith.visualization.real_time_visualizer import RealTimeVisualizer

# Type variable for generic function
F = TypeVar('F', bound=Callable[..., Any])

def profile_model(
        memory: bool = True,
        computation: bool = True,
        gpu: bool = False,
        save_report: bool = True,
        report_format: str = 'html',
        real_time_viz: bool = False
) -> Callable[[F], F]:
    """
    Decorator to profile a model's training or inference function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Async wrapper for the decorated function."""
            # Setup logging
            setup_logging(config.log_level)
            logger = logging.getLogger('memoraith')
            logger.info(f"Starting Memoraith Profiler for {func.__name__}...")

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
                logger.info(f"Initialized {adapter.__class__.__name__} for {model.__class__.__name__}")

                # Initialize visualizer if requested
                if real_time_viz:
                    visualizer = RealTimeVisualizer()
                    await visualizer.start()
                    logger.info("Real-time visualization started")

                # Start profiling the model
                await adapter.start_profiling()

                # Execute the wrapped function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)

                # Stop profiling
                await adapter.stop_profiling()

                # Get profiling data
                profiling_data = await adapter.get_profiling_data()

                # Update visualization if enabled
                if visualizer and profiling_data:
                    await visualizer.update(profiling_data)

                # Analyze results
                analyzer = Analyzer(profiling_data)
                analysis_results = await analyzer.run_analysis()
                logger.info("Analysis completed successfully")

                # Generate report if requested
                if save_report:
                    report_generator = ReportGenerator(analysis_results)
                    await report_generator.generate(format=report_format)
                    logger.info(f"Report generated in {config.output_path}")

                logger.info(f"Memoraith Profiling for {func.__name__} completed successfully")
                return result

            except MemoraithError as e:
                logger.error(f"MemoraithError: {str(e)}")
                raise
            except Exception as e:
                logger.exception(f"An unexpected error occurred during profiling: {str(e)}")
                raise
            finally:
                if visualizer:
                    try:
                        await visualizer.stop()
                        logger.info("Real-time visualization stopped")
                    except Exception as e:
                        logger.error(f"Error stopping visualizer: {str(e)}")

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Synchronous wrapper that runs the async wrapper."""
            return asyncio.run(async_wrapper(*args, **kwargs))

        # Use the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator

def set_output_path(path: str) -> None:
    """
    Set the output path for profiling reports.
    """
    try:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        config.output_path = output_path
        logging.getLogger('memoraith').info(f"Output path set to: {output_path}")
    except Exception as e:
        raise MemoraithError(f"Failed to set output path: {str(e)}")