"""
TensorFlow integration adapter for Memoraith profiler.
Enterprise-grade implementation with comprehensive error handling,
performance optimization, and cross-version compatibility.
"""
import numpy as np
import tensorflow as tf
import time
import logging
import asyncio
import sys
import os
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path

from ..exceptions import MemoraithError, FrameworkNotSupportedError, GPUNotAvailableError
from .framework_adapter import FrameworkAdapter
from ..data_collection import TimeTracker
from ..config import config

# Defensive imports with version compatibility
try:
    from ..data_collection.cpu_memory import CPUMemoryTracker
    CPU_TRACKER_AVAILABLE = True
except ImportError:
    CPU_TRACKER_AVAILABLE = False

try:
    from ..data_collection.gpu_memory import GPUMemoryTracker
    GPU_TRACKER_AVAILABLE = True
except ImportError:
    GPU_TRACKER_AVAILABLE = False

# TensorFlow version compatibility handling
TF_VERSION = tuple(map(int, tf.__version__.split('.')[:2]))
TF2 = TF_VERSION[0] >= 2

# Safe Keras import with TensorFlow version handling
try:
    if TF2:
        # For TensorFlow 2.x
        keras = tf.keras
    else:
        # Fallback for older versions
        import keras
    KERAS_AVAILABLE = True
except ImportError:
    keras = None
    KERAS_AVAILABLE = False

class TensorFlowAdapter(FrameworkAdapter):
    """
    Enterprise-grade adapter for TensorFlow model profiling with enhanced
    reliability, comprehensive metrics collection, and cross-version compatibility.

    Features:
    - Robust error handling for all operations
    - Compatibility with TensorFlow 1.x and 2.x
    - Graceful degradation for missing dependencies
    - Comprehensive instrumentation and metrics collection
    - Thread-safe and async-compatible operations
    """

    def __init__(self, model: Any):
        """
        Initialize the TensorFlow adapter with comprehensive error handling
        and resource validation.

        Args:
            model: The TensorFlow model to profile (Keras model or custom model)
        """
        super().__init__(model)

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self._is_profiling = False

        # Initialize tracking components with defensive programming
        self.time_tracker = TimeTracker()

        # CPU tracker initialization with error handling
        self.cpu_tracker = None
        if CPU_TRACKER_AVAILABLE:
            try:
                self.cpu_tracker = CPUMemoryTracker(detailed=False)  # Disable detailed to avoid attribute errors
                self.logger.debug("CPU memory tracker initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CPU memory tracker: {str(e)}")
        else:
            self.logger.info("CPU memory tracking unavailable - missing dependencies")

        # GPU tracker initialization with capability detection
        self.gpu_tracker = None
        if GPU_TRACKER_AVAILABLE and hasattr(tf, 'test') and hasattr(tf.test, 'is_built_with_cuda'):
            try:
                if tf.test.is_built_with_cuda() and config.enable_gpu:
                    self.gpu_tracker = GPUMemoryTracker()
                    self.logger.debug("GPU memory tracker initialized successfully")
                else:
                    self.logger.info("GPU profiling disabled: CUDA not available or not enabled in config")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU memory tracker: {str(e)}")
        else:
            self.logger.info("GPU memory tracking unavailable - missing dependencies or GPU support")

        # Cache for original methods
        self.original_call = None
        self._original_fit = None
        self._original_predict = None
        self._original_evaluate = None

        # Performance metrics
        self._profile_start_time = None
        self._tf_eager_mode = None

        # Check execution mode for TensorFlow
        if TF2:
            self._tf_eager_mode = tf.executing_eagerly()
            self.logger.debug(f"TensorFlow running in eager mode: {self._tf_eager_mode}")

        # Validate model initialization
        self._validate_model_initialization()

    def _validate_model_initialization(self) -> None:
        """
        Validates model initialization and logs diagnostic information
        to enable better troubleshooting of potential issues.
        """
        try:
            # Log model type information
            model_type = type(self.model).__name__
            self.logger.debug(f"TensorFlow model type: {model_type}")

            # Check for Keras Sequential model
            is_sequential = False
            if KERAS_AVAILABLE and hasattr(keras, 'Sequential'):
                is_sequential = isinstance(self.model, keras.Sequential)
                if is_sequential:
                    self.logger.debug(f"Model identified as Keras Sequential with {len(self.model.layers)} layers")

            # Check for Keras Functional model
            is_functional = False
            if KERAS_AVAILABLE and hasattr(keras, 'Model'):
                is_functional = isinstance(self.model, keras.Model) and not is_sequential
                if is_functional and hasattr(self.model, 'layers'):
                    self.logger.debug(f"Model identified as Keras Functional with {len(self.model.layers)} layers")

            # Check for custom model with call method
            if hasattr(self.model, 'call') and callable(self.model.call):
                self.logger.debug("Model has custom call method - compatible with TensorFlow profiling")

            # Check if model is compiled (for Keras models)
            if hasattr(self.model, 'compiled_loss'):
                if self.model.compiled_loss is not None:
                    self.logger.debug("Model is compiled with loss function")
                else:
                    self.logger.warning("Model has not been compiled - some metrics may be unavailable")

            # Check for custom training step
            if hasattr(self.model, 'train_step') and callable(self.model.train_step):
                self.logger.debug("Model has custom train_step method")

        except Exception as e:
            self.logger.warning(f"Model validation failed (non-critical): {str(e)}")

    def log_profiling_start(self) -> None:
        """
        Logs profiling start event with diagnostic information.
        Provides visibility into the profiling process and setup.
        """
        self.logger.info("TensorFlow profiling started")

        # Log additional diagnostic information when debug logging is enabled
        if self.logger.isEnabledFor(logging.DEBUG):
            if TF2:
                self.logger.debug(f"TensorFlow version: {tf.__version__} (TF2)")
                self.logger.debug(f"Eager execution: {self._tf_eager_mode}")
            else:
                self.logger.debug(f"TensorFlow version: {tf.__version__} (TF1)")

            # Log GPU information if available
            if hasattr(tf, 'config') and hasattr(tf.config, 'list_physical_devices'):
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    self.logger.debug(f"TensorFlow GPU devices: {len(gpus)}")
                    for i, gpu in enumerate(gpus):
                        self.logger.debug(f"  - GPU {i}: {gpu.name}")
                else:
                    self.logger.debug("TensorFlow running on CPU only")

    def log_profiling_stop(self) -> None:
        """
        Logs profiling stop event with duration and resource statistics.
        """
        duration = "unknown"
        if self._profile_start_time is not None:
            duration = f"{time.time() - self._profile_start_time:.2f} seconds"

        self.logger.info(f"TensorFlow profiling stopped (duration: {duration})")

    async def start_profiling(self) -> None:
        """
        Start profiling TensorFlow model with comprehensive error handling.

        This implementation handles all edge cases, ensures proper resource
        initialization, and provides graceful degradation when components fail.

        Raises:
            MemoraithError: On critical failures that prevent profiling
        """
        if self._is_profiling:
            self.logger.warning("Profiling is already active - ignoring duplicate start request")
            return

        try:
            # Mark profiling as started to prevent duplicate calls and ensure cleanup
            self._is_profiling = True
            self._profile_start_time = time.time()

            # Cache original methods with defensive programming
            if hasattr(self.model, 'call') and callable(self.model.call):
                self.original_call = self.model.call
                self.model.call = self._wrapped_call
                self.logger.debug("Instrumented model.call method for profiling")

            # Handle various TensorFlow model types and methods
            if hasattr(self.model, 'fit') and callable(self.model.fit):
                self._original_fit = self.model.fit
                self.model.fit = self._wrapped_fit
                self.logger.debug("Instrumented model.fit method for profiling")

            if hasattr(self.model, 'predict') and callable(self.model.predict):
                self._original_predict = self.model.predict
                self.model.predict = self._wrapped_predict
                self.logger.debug("Instrumented model.predict method for profiling")

            if hasattr(self.model, 'evaluate') and callable(self.model.evaluate):
                self._original_evaluate = self.model.evaluate
                self.model.evaluate = self._wrapped_evaluate
                self.logger.debug("Instrumented model.evaluate method for profiling")

            # Start resource monitoring with robust error handling
            # Start CPU memory tracking if available
            if self.cpu_tracker is not None:
                try:
                    # Use synchronous API for CPU tracker to avoid async errors
                    self.cpu_tracker.start()
                    self.logger.debug("CPU memory tracking started")
                except Exception as e:
                    self.logger.warning(f"CPU memory tracking failed to start: {str(e)}")
                    # Continue execution - non-critical component
            else:
                self.logger.debug("CPU memory tracking not available - skipping")

            # Start GPU memory tracking if available
            if self.gpu_tracker is not None:
                try:
                    # Handle both sync and async implementations
                    if hasattr(self.gpu_tracker, 'start') and asyncio.iscoroutinefunction(self.gpu_tracker.start):
                        await self.gpu_tracker.start()
                    elif hasattr(self.gpu_tracker, 'start'):
                        self.gpu_tracker.start()
                    self.logger.debug("GPU memory tracking started")
                except Exception as e:
                    self.logger.warning(f"GPU memory tracking failed to start: {str(e)}")
                    # Continue execution - non-critical component
            else:
                self.logger.debug("GPU memory tracking not available - skipping")

            # Log initial memory usage
            initial_memory = await self._get_initial_memory()
            self.data['initial_memory'] = initial_memory
            self.logger.debug(f"Initial memory: CPU={initial_memory.get('cpu_memory', 0):.2f}MB, GPU={initial_memory.get('gpu_memory', 0):.2f}MB")

            # Log profiling start with diagnostic information
            self.log_profiling_start()

        except Exception as e:
            # Ensure comprehensive error reporting with traceback
            error_msg = f"Error starting TensorFlow profiling: {str(e)}"
            stack_trace = traceback.format_exc()
            self.logger.error(f"{error_msg}\n{stack_trace}")

            # Always clean up on error to prevent resource leaks
            await self.cleanup()

            # Ensure we're not marked as profiling if initialization failed
            self._is_profiling = False

            # Propagate error with context
            raise MemoraithError(f"Failed to start TensorFlow profiling: {str(e)}")

    async def stop_profiling(self) -> None:
        """
        Stop profiling and restore the original model methods with
        comprehensive resource cleanup and error handling.

        Implements safe resource release to prevent memory leaks and
        ensure model integrity after profiling completes or fails.
        """
        if not self._is_profiling:
            self.logger.warning("Profiling is not active - ignoring stop request")
            return

        try:
            # Mark as not profiling to prevent duplicate cleanup
            self._is_profiling = False

            # Restore original model methods with defensive programming
            if self.original_call is not None:
                try:
                    self.model.call = self.original_call
                    self.original_call = None
                    self.logger.debug("Restored original model.call method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.call: {str(e)}")

            if self._original_fit is not None:
                try:
                    self.model.fit = self._original_fit
                    self._original_fit = None
                    self.logger.debug("Restored original model.fit method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.fit: {str(e)}")

            if self._original_predict is not None:
                try:
                    self.model.predict = self._original_predict
                    self._original_predict = None
                    self.logger.debug("Restored original model.predict method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.predict: {str(e)}")

            if self._original_evaluate is not None:
                try:
                    self.model.evaluate = self._original_evaluate
                    self._original_evaluate = None
                    self.logger.debug("Restored original model.evaluate method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.evaluate: {str(e)}")

            # Stop resource tracking with robust error handling
            # Stop CPU memory tracking
            if self.cpu_tracker is not None:
                try:
                    # Use the appropriate API based on availability
                    if hasattr(self.cpu_tracker, 'stop'):
                        # Execute synchronously to avoid async errors
                        self.cpu_tracker.stop()
                        self.logger.debug("CPU memory tracking stopped")
                except Exception as e:
                    self.logger.warning(f"Error stopping CPU memory tracking: {str(e)}")

            # Stop GPU memory tracking
            if self.gpu_tracker is not None:
                try:
                    # Handle both sync and async implementations
                    if hasattr(self.gpu_tracker, 'stop') and asyncio.iscoroutinefunction(self.gpu_tracker.stop):
                        await self.gpu_tracker.stop()
                    elif hasattr(self.gpu_tracker, 'stop'):
                        self.gpu_tracker.stop()
                    self.logger.debug("GPU memory tracking stopped")
                except Exception as e:
                    self.logger.warning(f"Error stopping GPU memory tracking: {str(e)}")

            # Collect final memory metrics
            try:
                final_memory = await self._get_final_memory()
                self.data['final_memory'] = final_memory
                self.logger.debug(f"Final memory: CPU={final_memory.get('cpu_memory', 0):.2f}MB, GPU={final_memory.get('gpu_memory', 0):.2f}MB")

                # Calculate memory delta if initial memory was captured
                if 'initial_memory' in self.data:
                    initial = self.data['initial_memory']
                    memory_delta = {
                        'cpu_memory': final_memory.get('cpu_memory', 0) - initial.get('cpu_memory', 0),
                        'gpu_memory': final_memory.get('gpu_memory', 0) - initial.get('gpu_memory', 0)
                    }
                    self.data['memory_delta'] = memory_delta
                    self.logger.debug(f"Memory delta: CPU={memory_delta['cpu_memory']:.2f}MB, GPU={memory_delta['gpu_memory']:.2f}MB")
            except Exception as e:
                self.logger.warning(f"Error collecting final memory metrics: {str(e)}")

            # Log profiling completion
            self.log_profiling_stop()

        except Exception as e:
            # Log error but don't re-raise to ensure cleanup completes
            self.logger.error(f"Error during profiling shutdown: {str(e)}", exc_info=True)

    def _wrapped_call(self, *args: Any, **kwargs: Any) -> Any:
        """
        Instrumented wrapper for model.call method to collect performance metrics.

        Args:
            *args: Positional arguments to pass to the original call method
            **kwargs: Keyword arguments to pass to the original call method

        Returns:
            The result of the original call method
        """
        if not self._is_profiling or self.original_call is None:
            # Fail safe if profiling is disabled or original method not available
            if hasattr(self.model, 'call') and callable(self.model.call):
                return self.model.call(*args, **kwargs)
            else:
                raise AttributeError("Model does not have a valid call method")

        try:
            # Start metrics collection
            call_id = f"call_{len([k for k in self.data.keys() if k.startswith('call_')])}"
            self.time_tracker.start(call_id)

            # Execute original method
            result = self.original_call(*args, **kwargs)

            # Collect metrics
            self.time_tracker.stop(call_id)
            execution_time = self.time_tracker.get_duration(call_id)

            # Add execution data to profiling results
            self.data[call_id] = {
                'time': execution_time,
                'timestamp': time.time(),
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            }

            # Add input/output shape info if available
            try:
                if args and hasattr(args[0], 'shape'):
                    self.data[call_id]['input_shape'] = tuple(args[0].shape)
                if hasattr(result, 'shape'):
                    self.data[call_id]['output_shape'] = tuple(result.shape)
            except Exception as shape_error:
                self.logger.debug(f"Could not determine shapes: {str(shape_error)}")

            return result

        except Exception as e:
            self.logger.error(f"Error in wrapped call: {str(e)}")
            # Fall back to original method on error
            if self.original_call is not None:
                return self.original_call(*args, **kwargs)
            raise

    def _wrapped_fit(self, *args: Any, **kwargs: Any) -> Any:
        """
        Instrumented wrapper for model.fit method to collect training metrics.

        Provides comprehensive monitoring of the training process with callbacks
        integration for epoch-level metrics collection.

        Args:
            *args: Positional arguments to the original fit method
            **kwargs: Keyword arguments to the original fit method

        Returns:
            The result of the original fit method with added profiling
        """
        if not self._is_profiling or self._original_fit is None:
            # Fail safe if profiling is disabled or original method not available
            if hasattr(self.model, 'fit') and callable(self.model.fit):
                return self.model.fit(*args, **kwargs)
            else:
                raise AttributeError("Model does not have a valid fit method")

        try:
            # Track overall training time
            self.time_tracker.start('model_fit')

            # Add profiling callback to monitor training progress
            callbacks = list(kwargs.get('callbacks', []) or [])
            monitoring_callback = self._create_monitoring_callback()
            callbacks.append(monitoring_callback)
            kwargs['callbacks'] = callbacks

            # Call original fit method
            result = self._original_fit(*args, **kwargs)

            # Stop timing and collect metrics
            self.time_tracker.stop('model_fit')
            total_time = self.time_tracker.get_duration('model_fit')

            # Extract history metrics if available
            history_data = {}
            if hasattr(result, 'history'):
                # Convert all history values to Python native types for serialization
                history_data = {
                    k: [float(v) if isinstance(v, (int, float)) else v for v in vals]
                    for k, vals in result.history.items()
                }

            # Store comprehensive training metrics
            self.data['training'] = {
                'total_time': total_time,
                'epochs': kwargs.get('epochs', len(history_data.get('loss', []))),
                'batch_size': kwargs.get('batch_size', None),
                'samples': len(args[0]) if args and hasattr(args[0], '__len__') else None,
                'history': history_data,
                'timestamp': time.time()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in wrapped fit: {str(e)}")
            # Fall back to original method on error
            if self._original_fit is not None:
                return self._original_fit(*args, **kwargs)
            raise

    def _wrapped_predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Instrumented wrapper for model.predict method to collect inference metrics.

        Args:
            *args: Positional arguments to the original predict method
            **kwargs: Keyword arguments to the original predict method

        Returns:
            The result of the original predict method with added profiling
        """
        if not self._is_profiling or self._original_predict is None:
            # Fail safe if profiling is disabled or original method not available
            if hasattr(self.model, 'predict') and callable(self.model.predict):
                return self.model.predict(*args, **kwargs)
            else:
                raise AttributeError("Model does not have a valid predict method")

        try:
            # Start prediction timing
            predict_id = f"predict_{len([k for k in self.data.keys() if k.startswith('predict_')])}"
            self.time_tracker.start(predict_id)

            # Execute original method
            result = self._original_predict(*args, **kwargs)

            # Stop timing and collect metrics
            self.time_tracker.stop(predict_id)
            execution_time = self.time_tracker.get_duration(predict_id)

            # Determine batch size and shapes if possible
            batch_size = kwargs.get('batch_size', None)
            if batch_size is None and args and hasattr(args[0], '__len__'):
                batch_size = len(args[0])

            # Collect shapes if available
            input_shape = None
            output_shape = None
            if args and hasattr(args[0], 'shape'):
                input_shape = tuple(args[0].shape)
            if hasattr(result, 'shape'):
                output_shape = tuple(result.shape)

            # Store prediction metrics
            self.data[predict_id] = {
                'time': execution_time,
                'batch_size': batch_size,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'timestamp': time.time(),
                'throughput': batch_size / execution_time if batch_size and execution_time > 0 else None
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in wrapped predict: {str(e)}")
            # Fall back to original method on error
            if self._original_predict is not None:
                return self._original_predict(*args, **kwargs)
            raise

    def _wrapped_evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Instrumented wrapper for model.evaluate method to collect evaluation metrics.

        Args:
            *args: Positional arguments to the original evaluate method
            **kwargs: Keyword arguments to the original evaluate method

        Returns:
            The result of the original evaluate method with added profiling
        """
        if not self._is_profiling or self._original_evaluate is None:
            # Fail safe if profiling is disabled or original method not available
            if hasattr(self.model, 'evaluate') and callable(self.model.evaluate):
                return self.model.evaluate(*args, **kwargs)
            else:
                raise AttributeError("Model does not have a valid evaluate method")

        try:
            # Start evaluation timing
            eval_id = f"evaluate_{len([k for k in self.data.keys() if k.startswith('evaluate_')])}"
            self.time_tracker.start(eval_id)

            # Execute original method
            result = self._original_evaluate(*args, **kwargs)

            # Stop timing and collect metrics
            self.time_tracker.stop(eval_id)
            execution_time = self.time_tracker.get_duration(eval_id)

            # Format metrics based on result type
            metrics_dict = {}

            # Handle different return types from evaluate
            if isinstance(result, list):
                # Multiple metrics returned as a list
                metric_names = []
                if hasattr(self.model, 'metrics_names'):
                    metric_names = self.model.metrics_names
                else:
                    # Create default names if metrics_names not available
                    metric_names = [f'metric_{i}' for i in range(len(result))]

                # Create dictionary mapping metric names to values
                metrics_dict = {
                    name: float(value) for i, (name, value) in enumerate(zip(metric_names, result))
                    if i < len(result)  # Handle case where fewer values than names
                }
            elif isinstance(result, (int, float)):
                # Single scalar result (typically loss)
                metrics_dict = {'loss': float(result)}
            elif hasattr(result, 'numpy'):
                # TensorFlow eager tensor
                try:
                    value = result.numpy()
                    metrics_dict = {'result': float(value) if hasattr(value, 'item') else value}
                except Exception as tensor_error:
                    self.logger.debug(f"Could not convert tensor result: {str(tensor_error)}")
                    metrics_dict = {'result': 'tensor_conversion_error'}

            # Collect dataset size if available
            dataset_size = None
            if args and hasattr(args[0], '__len__'):
                dataset_size = len(args[0])

            # Store evaluation metrics
            self.data[eval_id] = {
                'time': execution_time,
                'metrics': metrics_dict,
                'dataset_size': dataset_size,
                'batch_size': kwargs.get('batch_size', None),
                'timestamp': time.time()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in wrapped evaluate: {str(e)}")
            # Fall back to original method on error
            if self._original_evaluate is not None:
                return self._original_evaluate(*args, **kwargs)
            raise

    def _create_monitoring_callback(self) -> Any:
        """
        Create a TensorFlow callback for detailed training process monitoring.

        Returns:
            keras.callbacks.Callback: Custom monitoring callback
        """
        if not KERAS_AVAILABLE or not hasattr(keras, 'callbacks'):
            # Create minimal callback implementation if Keras not available
            return None

        adapter = self  # Reference for inner class

        class ProfilingCallback(keras.callbacks.Callback):
            """Custom callback for comprehensive training process monitoring."""

            def __init__(self):
                super().__init__()
                self.epoch_times = []
                self.batch_times = []
                self.current_epoch_start = None
                self.current_batch_start = None
                self.memory_snapshots = []

            def on_train_begin(self, logs=None):
                """Record training start metrics."""
                adapter.logger.debug("Training started")
                # Capture memory snapshot at training start
                self._capture_memory_snapshot('train_begin')

            def on_train_end(self, logs=None):
                """Record training end metrics."""
                adapter.logger.debug("Training completed")
                # Capture memory snapshot at training end
                self._capture_memory_snapshot('train_end')

                # Add overall epoch statistics to adapter data
                if self.epoch_times:
                    adapter.data['epoch_stats'] = {
                        'count': len(self.epoch_times),
                        'total_time': sum(self.epoch_times),
                        'avg_time': sum(self.epoch_times) / len(self.epoch_times),
                        'min_time': min(self.epoch_times),
                        'max_time': max(self.epoch_times)
                    }

            def on_epoch_begin(self, epoch, logs=None):
                """Record epoch start time and metrics."""
                self.current_epoch_start = time.time()
                adapter.logger.debug(f"Starting epoch {epoch+1}")

            def on_epoch_end(self, epoch, logs=None):
                """Record epoch metrics including duration and memory usage."""
                if self.current_epoch_start:
                    epoch_time = time.time() - self.current_epoch_start
                    self.epoch_times.append(epoch_time)

                    # Capture memory snapshot at epoch end
                    snapshot_data = self._capture_memory_snapshot(f'epoch_{epoch}_end')

                    # Create metrics dict with native Python types for serialization
                    metrics = {}
                    if logs:
                        metrics = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in logs.items()}

                    # Record epoch metrics in adapter data
                    adapter.data[f'epoch_{epoch}'] = {
                        'index': epoch,
                        'time': epoch_time,
                        'metrics': metrics,
                        'memory': snapshot_data
                    }

                    adapter.logger.debug(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

            def on_batch_begin(self, batch, logs=None):
                """Record batch start time."""
                self.current_batch_start = time.time()

            def on_batch_end(self, batch, logs=None):
                """Record batch metrics including duration."""
                if self.current_batch_start:
                    batch_time = time.time() - self.current_batch_start
                    self.batch_times.append(batch_time)

                    # Only record detailed batch metrics if we have few batches
                    # to avoid excessive memory usage
                    if len(self.batch_times) <= 100:  # Limit detailed batch recording
                        metrics = {}
                        if logs:
                            metrics = {k: float(v) if isinstance(v, (int, float)) else v
                                       for k, v in logs.items()}

                        # Record batch metrics in adapter data
                        batch_key = f'batch_{len(self.batch_times)-1}'
                        adapter.data[batch_key] = {
                            'index': batch,
                            'time': batch_time,
                            'metrics': metrics
                        }

            def _capture_memory_snapshot(self, label: str) -> Dict[str, float]:
                """
                Capture current memory usage with both CPU and GPU metrics.

                Args:
                    label: Identifier for this memory snapshot

                Returns:
                    Dict[str, float]: Memory metrics or empty dict on error
                """
                snapshot = {'timestamp': time.time(), 'label': label}

                # Get CPU memory if tracker available
                if adapter.cpu_tracker is not None and hasattr(adapter.cpu_tracker, 'get_peak_memory'):
                    try:
                        cpu_memory = adapter.cpu_tracker.get_peak_memory()
                        if isinstance(cpu_memory, dict) and 'rss' in cpu_memory:
                            snapshot['cpu_memory'] = cpu_memory['rss']
                    except Exception as e:
                        adapter.logger.debug(f"Error capturing CPU memory: {str(e)}")

                # Get GPU memory if tracker available (don't use await in callback)
                # We just skip GPU memory tracking in callbacks

                # Store snapshot
                self.memory_snapshots.append(snapshot)
                return snapshot

        return ProfilingCallback()

    async def _get_initial_memory(self) -> Dict[str, float]:
        """
        Get initial memory usage before profiling with robust error handling.

        Returns:
            Dict[str, float]: Memory usage statistics or empty values on error
        """
        cpu_memory = 0.0
        gpu_memory = 0.0

        # Safely get CPU memory
        if self.cpu_tracker is not None and hasattr(self.cpu_tracker, 'get_peak_memory'):
            try:
                cpu_data = self.cpu_tracker.get_peak_memory()
                if isinstance(cpu_data, dict) and 'rss' in cpu_data:
                    cpu_memory = cpu_data['rss']
            except Exception as e:
                self.logger.debug(f"Error getting initial CPU memory: {str(e)}")

        # Safely get GPU memory
        if self.gpu_tracker is not None:
            try:
                # Handle both async and sync implementations
                if hasattr(self.gpu_tracker, 'get_peak_memory'):
                    if asyncio.iscoroutinefunction(self.gpu_tracker.get_peak_memory):
                        gpu_memory = await self.gpu_tracker.get_peak_memory()
                    else:
                        gpu_memory = self.gpu_tracker.get_peak_memory()
            except Exception as e:
                self.logger.debug(f"Error getting initial GPU memory: {str(e)}")

        return {
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'timestamp': time.time()
        }

    async def _get_final_memory(self) -> Dict[str, float]:
        """
        Get final memory usage after profiling with robust error handling.

        Returns:
            Dict[str, float]: Memory usage statistics or empty values on error
        """
        cpu_memory = 0.0
        gpu_memory = 0.0

        # Safely get CPU memory
        if self.cpu_tracker is not None and hasattr(self.cpu_tracker, 'get_peak_memory'):
            try:
                cpu_data = self.cpu_tracker.get_peak_memory()
                if isinstance(cpu_data, dict) and 'rss' in cpu_data:
                    cpu_memory = cpu_data['rss']
            except Exception as e:
                self.logger.debug(f"Error getting final CPU memory: {str(e)}")

        # Safely get GPU memory
        if self.gpu_tracker is not None:
            try:
                # Handle both async and sync implementations
                if hasattr(self.gpu_tracker, 'get_peak_memory'):
                    if asyncio.iscoroutinefunction(self.gpu_tracker.get_peak_memory):
                        gpu_memory = await self.gpu_tracker.get_peak_memory()
                    else:
                        gpu_memory = self.gpu_tracker.get_peak_memory()
            except Exception as e:
                self.logger.debug(f"Error getting final GPU memory: {str(e)}")

        return {
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'timestamp': time.time()
        }

    async def profile_inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Profile the inference process for a single input with detailed metrics.

        Args:
            input_data: Input data for model inference

        Returns:
            Dict[str, Any]: Comprehensive profiling metrics for the inference operation

        Raises:
            MemoraithError: If profiling fails critically
        """
        # Track if we started profiling locally (for cleanup)
        local_profiling = False

        try:
            # Start profiling if not already active
            if not self._is_profiling:
                await self.start_profiling()
                local_profiling = True

            # Execute model inference with timing
            self.time_tracker.start('inference')

            # Execute inference based on model type
            if hasattr(self.model, '__call__'):
                output = self.model(input_data)
            else:
                self.logger.error("Model does not have a __call__ method")
                raise MemoraithError("Model is not callable for inference")

            # Stop timing
            self.time_tracker.stop('inference')
            inference_time = self.time_tracker.get_duration('inference')

            # Get current memory usage
            memory_snapshot = await self._get_current_memory_snapshot()

            # Build comprehensive profiling data
            profiling_data = {
                'inference_time': inference_time,
                'timestamp': time.time(),
                'memory': memory_snapshot,
                'input_shape': self._get_tensor_shape(input_data),
                'output_shape': self._get_tensor_shape(output)
            }

            # Calculate throughput if batch size is detectable
            if input_data is not None and hasattr(input_data, 'shape') and len(input_data.shape) > 0:
                batch_size = input_data.shape[0]
                profiling_data['batch_size'] = batch_size
                if inference_time > 0:
                    profiling_data['throughput'] = batch_size / inference_time  # samples/second

            # Store in instance data for API consistency
            profile_id = f"inference_{len([k for k in self.data.keys() if k.startswith('inference_')])}"
            self.data[profile_id] = profiling_data

            return profiling_data

        except Exception as e:
            self.logger.error(f"Error during inference profiling: {str(e)}", exc_info=True)
            raise MemoraithError(f"TensorFlow inference profiling failed: {str(e)}")

        finally:
            # Stop profiling if we started it locally
            if local_profiling and self._is_profiling:
                await self.stop_profiling()

    async def _get_current_memory_snapshot(self) -> Dict[str, float]:
        """
        Get current memory usage snapshot with comprehensive metrics.

        Returns:
            Dict[str, float]: Current memory usage across all tracked resources
        """
        cpu_memory = 0.0
        gpu_memory = 0.0

        # Get current CPU memory
        if self.cpu_tracker is not None:
            try:
                if hasattr(self.cpu_tracker, 'get_peak_memory'):
                    cpu_data = self.cpu_tracker.get_peak_memory()
                    if isinstance(cpu_data, dict) and 'rss' in cpu_data:
                        cpu_memory = cpu_data['rss']
            except Exception as e:
                self.logger.debug(f"Error getting current CPU memory: {str(e)}")

        # Get current GPU memory
        if self.gpu_tracker is not None:
            try:
                if hasattr(self.gpu_tracker, 'get_current_memory'):
                    if asyncio.iscoroutinefunction(self.gpu_tracker.get_current_memory):
                        gpu_memory = await self.gpu_tracker.get_current_memory()
                    else:
                        gpu_memory = self.gpu_tracker.get_current_memory()
            except Exception as e:
                self.logger.debug(f"Error getting current GPU memory: {str(e)}")

        # Get process-wide memory stats
        process_memory = {}
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            process_memory = {
                'rss': mem_info.rss / (1024 * 1024),  # MB
                'vms': mem_info.vms / (1024 * 1024)   # MB
            }
        except Exception as e:
            self.logger.debug(f"Error getting process memory: {str(e)}")

        return {
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'process': process_memory,
            'timestamp': time.time()
        }

    def _get_tensor_shape(self, tensor_data: Any) -> Optional[Tuple[int, ...]]:
        """
        Safely extract shape information from tensor-like objects.

        Args:
            tensor_data: Tensor, array, or other data structure

        Returns:
            Optional[Tuple[int, ...]]: Shape tuple or None if shape can't be determined
        """
        try:
            # Handle different types of tensor/array objects
            if hasattr(tensor_data, 'shape'):
                # TensorFlow tensor or numpy array
                if hasattr(tensor_data.shape, '__iter__'):
                    return tuple(tensor_data.shape)
                return tuple([tensor_data.shape])
            elif isinstance(tensor_data, (list, tuple)) and tensor_data:
                # Handle list of tensors
                if all(hasattr(t, 'shape') for t in tensor_data):
                    return tuple([self._get_tensor_shape(t) for t in tensor_data])
                # Try to infer shape from nested lists
                if isinstance(tensor_data[0], (list, tuple)):
                    return (len(tensor_data), len(tensor_data[0]))
                return (len(tensor_data),)
            # Shape cannot be determined
            return None
        except Exception as e:
            self.logger.debug(f"Error determining tensor shape: {str(e)}")
            return None

    async def profile_training_step(self,
                                    input_data: Any,
                                    target: Any,
                                    optimizer: Optional[Any] = None,
                                    loss: Optional[Any] = None) -> Dict[str, Any]:
        """
        Profile a single training step with detailed metrics collection.

        Args:
            input_data: Input data for training
            target: Target/label data for training
            optimizer: Optional optimizer (will use model's optimizer if None)
            loss: Optional loss function (will use model's loss if None)

        Returns:
            Dict[str, Any]: Comprehensive profiling metrics for the training step

        Raises:
            MemoraithError: If profiling fails critically
        """
        # Track if we started profiling locally (for cleanup)
        local_profiling = False

        try:
            # Start profiling if not already active
            if not self._is_profiling:
                await self.start_profiling()
                local_profiling = True

            # Verify model is compiled or optimizer provided
            if not optimizer and not hasattr(self.model, 'optimizer'):
                raise MemoraithError("Model must be compiled or optimizer must be provided")

            # Execute training step based on model structure
            self.time_tracker.start('training_step')

            # Track metrics for the training step
            metrics_result = {}

            # Different training approaches based on model type
            if hasattr(self.model, 'train_step'):
                # Custom training step is available - use it
                packed_data = self._pack_data(input_data, target)
                step_result = self.model.train_step(packed_data)

                # Extract metrics from step result
                if isinstance(step_result, dict):
                    metrics_result = {k: float(v) if hasattr(v, 'numpy') else v
                                      for k, v in step_result.items()}
                else:
                    metrics_result = {'step_result': step_result}

            else:
                # Use standard training step with GradientTape
                opt = optimizer or self.model.optimizer
                loss_fn = loss or self.model.compiled_loss

                # Execute training step with TensorFlow's recommended approach
                if TF2:  # TensorFlow 2.x with GradientTape
                    with tf.GradientTape() as tape:
                        predictions = self.model(input_data, training=True)
                        step_loss = loss_fn(target, predictions)

                    # Compute gradients and apply
                    gradients = tape.gradient(step_loss, self.model.trainable_variables)
                    opt.apply_gradients(zip(gradients, self.model.trainable_variables))

                    # Update metrics if available
                    if hasattr(self.model, 'compiled_metrics') and self.model.compiled_metrics is not None:
                        self.model.compiled_metrics.update_state(target, predictions)
                        for metric in self.model.metrics:
                            metrics_result[metric.name] = float(metric.result().numpy())

                    # Always include loss
                    metrics_result['loss'] = float(step_loss.numpy()) if hasattr(step_loss, 'numpy') else float(step_loss)

                else:  # TensorFlow 1.x or other approaches
                    self.logger.warning("Using fallback training approach - metrics may be limited")
                    # Execute a training step using the model's defined training operation
                    # This is highly model-specific and may not work for all TF1 models
                    result = self.model(input_data, training=True)
                    if isinstance(result, tuple) and len(result) > 1:
                        predictions, loss_value = result
                    else:
                        predictions = result
                        loss_value = None

                    metrics_result['step_completed'] = True
                    if loss_value is not None:
                        metrics_result['loss'] = float(loss_value)

            # Stop timing
            self.time_tracker.stop('training_step')
            step_time = self.time_tracker.get_duration('training_step')

            # Get current memory usage
            memory_snapshot = await self._get_current_memory_snapshot()

            # Build comprehensive profiling data
            training_data = {
                'step_time': step_time,
                'timestamp': time.time(),
                'memory': memory_snapshot,
                'metrics': metrics_result,
                'input_shape': self._get_tensor_shape(input_data),
                'target_shape': self._get_tensor_shape(target)
            }

            # Add batch size info if available
            if input_data is not None and hasattr(input_data, 'shape') and len(input_data.shape) > 0:
                batch_size = input_data.shape[0]
                training_data['batch_size'] = batch_size
                if step_time > 0:
                    training_data['samples_per_second'] = batch_size / step_time

            # Store in the instance data for API consistency
            step_id = f"train_step_{len([k for k in self.data.keys() if k.startswith('train_step_')])}"
            self.data[step_id] = training_data

            return training_data

        except Exception as e:
            self.logger.error(f"Error during training step profiling: {str(e)}", exc_info=True)
            raise MemoraithError(f"TensorFlow training step profiling failed: {str(e)}")

        finally:
            # Stop profiling if we started it locally
            if local_profiling and self._is_profiling:
                await self.stop_profiling()

    def _pack_data(self, x: Any, y: Any) -> Any:
        """
        Pack input and target data as expected by model.train_step.

        Args:
            x: Input data
            y: Target data

        Returns:
            Properly formatted data for train_step
        """
        if isinstance(x, dict) and y is None:
            return x  # Data already in expected format

        # Handle TF1 vs TF2 differences
        if TF2:
            # TF2 typically expects tuple of (inputs, targets, sample_weight)
            return (x, y, None)

        return (x, y)  # Basic tuple of inputs and targets

    async def get_model_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive model architecture summary with detailed metrics.

        Returns:
            Dict[str, Any]: Detailed model architecture information
        """
        summary = {}

        try:
            # Get model type information
            summary['model_type'] = type(self.model).__name__

            # Determine framework version
            summary['tensorflow_version'] = tf.__version__
            summary['eager_mode'] = tf.executing_eagerly() if TF2 else False

            # Get basic model info for Keras models
            if KERAS_AVAILABLE:
                # Check if it's a Keras model
                is_keras_model = isinstance(self.model, keras.Model)
                summary['is_keras_model'] = is_keras_model

                if is_keras_model:
                    # Get parameter counts
                    if hasattr(self.model, 'count_params'):
                        total_params = self.model.count_params()
                        trainable_params = sum(keras.backend.count_params(w) for w in self.model.trainable_weights)
                        non_trainable_params = total_params - trainable_params

                        summary['total_params'] = int(total_params)
                        summary['trainable_params'] = int(trainable_params)
                        summary['non_trainable_params'] = int(non_trainable_params)

                    # Get input and output shapes
                    if hasattr(self.model, 'input_shape'):
                        summary['input_shape'] = list(self.model.input_shape)

                    if hasattr(self.model, 'output_shape'):
                        summary['output_shape'] = list(self.model.output_shape)

                    # Check compilation status
                    summary['is_compiled'] = hasattr(self.model, 'optimizer') and self.model.optimizer is not None

                    # Get detailed layer information
                    if hasattr(self.model, 'layers'):
                        layers_info = []
                        for i, layer in enumerate(self.model.layers):
                            layer_info = {
                                'index': i,
                                'name': layer.name,
                                'type': layer.__class__.__name__,
                                'trainable': layer.trainable
                            }

                            # Get parameter count if available
                            if hasattr(layer, 'count_params'):
                                layer_info['params'] = int(layer.count_params())

                            # Get input/output shapes if available
                            if hasattr(layer, 'input_shape'):
                                layer_info['input_shape'] = list(layer.input_shape) if layer.input_shape else None

                            if hasattr(layer, 'output_shape'):
                                layer_info['output_shape'] = list(layer.output_shape) if layer.output_shape else None

                            layers_info.append(layer_info)

                        summary['layers'] = layers_info
                        summary['layer_count'] = len(layers_info)

                    # Get optimizer information if compiled
                    if hasattr(self.model, 'optimizer') and self.model.optimizer:
                        optimizer = self.model.optimizer
                        optimizer_config = {}

                        # Extract optimizer configuration safely
                        if hasattr(optimizer, 'get_config'):
                            try:
                                raw_config = optimizer.get_config()
                                # Convert to serializable format
                                optimizer_config = {
                                    k: float(v) if isinstance(v, (int, float)) or hasattr(v, 'numpy') else str(v)
                                    for k, v in raw_config.items()
                                }
                            except Exception as e:
                                self.logger.debug(f"Error getting optimizer config: {str(e)}")
                                optimizer_config = {'error': str(e)}

                        summary['optimizer'] = {
                            'name': optimizer.__class__.__name__,
                            'learning_rate': float(keras.backend.get_value(optimizer.lr)) if hasattr(optimizer, 'lr') else None,
                            'parameters': optimizer_config
                        }

                    # Get loss function info
                    if hasattr(self.model, 'loss') and self.model.loss:
                        loss = self.model.loss
                        if callable(loss):
                            summary['loss'] = {'name': loss.__name__ if hasattr(loss, '__name__') else str(loss)}
                        elif isinstance(loss, str):
                            summary['loss'] = {'name': loss}
                        else:
                            summary['loss'] = {'name': loss.__class__.__name__ if hasattr(loss, '__class__') else str(loss)}

            # Calculate model size estimation
            if hasattr(self.model, 'weights') and self.model.weights:
                # Determine parameter byte size based on dtype
                dtype_size = 4  # Default: float32 = 4 bytes
                if self.model.weights and hasattr(self.model.weights[0], 'dtype'):
                    w_dtype = self.model.weights[0].dtype
                    if 'float16' in str(w_dtype) or 'half' in str(w_dtype):
                        dtype_size = 2
                    elif 'double' in str(w_dtype) or 'float64' in str(w_dtype):
                        dtype_size = 8
                    elif 'int8' in str(w_dtype) or 'uint8' in str(w_dtype):
                        dtype_size = 1

                # Count parameters with handling for different model types
                param_count = 0
                if KERAS_AVAILABLE and hasattr(keras.backend, 'count_params'):
                    try:
                        param_count = sum(keras.backend.count_params(w) for w in self.model.weights)
                    except Exception:
                        # Fallback to manual counting
                        for w in self.model.weights:
                            if hasattr(w, 'shape'):
                                param_count += int(np.prod(w.shape))
                else:
                    # Manual counting for non-Keras models
                    for w in self.model.weights:
                        if hasattr(w, 'shape'):
                            param_count += int(np.prod(w.shape))

                # Calculate size estimations
                size_bytes = param_count * dtype_size
                summary['model_size'] = {
                    'parameters': int(param_count),
                    'bytes': size_bytes,
                    'kilobytes': size_bytes / 1024,
                    'megabytes': size_bytes / (1024 * 1024),
                    'parameter_size': f"{dtype_size} bytes ({str(self.model.weights[0].dtype) if self.model.weights else 'unknown'})"
                }

            # Get device placement information
            device_info = {}
            if hasattr(tf, 'distribute') and hasattr(tf.distribute, 'get_strategy'):
                try:
                    strategy = tf.distribute.get_strategy()
                    device_info['distribution_strategy'] = strategy.__class__.__name__
                    if hasattr(strategy, 'extended') and hasattr(strategy.extended, 'worker_devices'):
                        device_info['worker_devices'] = [str(d) for d in strategy.extended.worker_devices]
                except Exception as e:
                    self.logger.debug(f"Error getting distribution strategy: {str(e)}")

            # Get current device placement for variables
            if self.model.weights:
                try:
                    device_info['variable_device'] = str(self.model.weights[0].device)
                except Exception:
                    device_info['variable_device'] = 'unknown'

            summary['device_info'] = device_info

            return summary

        except Exception as e:
            self.logger.error(f"Error generating model summary: {str(e)}", exc_info=True)
            return {'error': str(e), 'traceback': traceback.format_exc()}

    async def validate_model(self) -> bool:
        """
        Validate the TensorFlow model for profiling compatibility.

        Performs comprehensive checks to ensure the model can be properly profiled
        and identifies potential issues before profiling begins.

        Returns:
            bool: True if model is valid for profiling, False otherwise
        """
        try:
            # Check if it's a proper TensorFlow model
            if not hasattr(self.model, 'call') and not hasattr(self.model, '__call__'):
                self.logger.error("Model does not have a 'call' or '__call__' method")
                return False

            # Verify model has variables/weights
            has_weights = False
            if hasattr(self.model, 'weights'):
                has_weights = bool(self.model.weights)
            elif hasattr(self.model, 'variables'):
                has_weights = bool(self.model.variables)
            elif hasattr(self.model, 'trainable_variables'):
                has_weights = bool(self.model.trainable_variables)

            if not has_weights:
                self.logger.warning("Model does not have weights/variables (not built or empty model)")
                # Not necessarily a failure, but worth logging

            # Check if the model has been compiled
            is_compiled = False
            if hasattr(self.model, 'compiled_loss'):
                is_compiled = self.model.compiled_loss is not None
            elif hasattr(self.model, 'optimizer'):
                is_compiled = self.model.optimizer is not None

            if not is_compiled:
                self.logger.warning("Model has not been compiled - some profiling features may be limited")
                # Not a failure, but some profiling features will be limited

            # Check for TensorFlow 1.x models (different behavior)
            if not TF2:
                self.logger.warning("Using TensorFlow 1.x - some profiling features may be limited")

                # Additional checks for TF1 models
                if hasattr(tf, 'Session') and tf.Session is not None:
                    # Check for an active session
                    if not tf.get_default_session():
                        self.logger.warning("No default TensorFlow session found - some operations may fail")

            # If we've reached this point, basic validation passes
            return True

        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """
        Clean up resources used during profiling with comprehensive error handling.

        Implements thorough resource cleanup to prevent memory leaks and ensure
        the model and TensorFlow environment are properly restored.
        """
        try:
            self.logger.debug("Starting TensorFlow adapter cleanup")

            # Restore original methods if not already done
            if self.original_call is not None:
                try:
                    self.model.call = self.original_call
                    self.original_call = None
                    self.logger.debug("Restored original model.call method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.call: {str(e)}")

            if self._original_fit is not None:
                try:
                    self.model.fit = self._original_fit
                    self._original_fit = None
                    self.logger.debug("Restored original model.fit method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.fit: {str(e)}")

            if self._original_predict is not None:
                try:
                    self.model.predict = self._original_predict
                    self._original_predict = None
                    self.logger.debug("Restored original model.predict method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.predict: {str(e)}")

            if self._original_evaluate is not None:
                try:
                    self.model.evaluate = self._original_evaluate
                    self._original_evaluate = None
                    self.logger.debug("Restored original model.evaluate method")
                except Exception as e:
                    self.logger.warning(f"Error restoring model.evaluate: {str(e)}")

            # Stop resource tracking
            if self.cpu_tracker is not None:
                try:
                    if hasattr(self.cpu_tracker, 'stop'):
                        # Use synchronous method
                        self.cpu_tracker.stop()
                    self.logger.debug("CPU tracker stopped")
                except Exception as e:
                    self.logger.warning(f"Error stopping CPU tracker: {str(e)}")

            if self.gpu_tracker is not None:
                try:
                    if hasattr(self.gpu_tracker, 'stop'):
                        if asyncio.iscoroutinefunction(self.gpu_tracker.stop):
                            # Use event loop if available
                            if asyncio.get_event_loop().is_running():
                                await self.gpu_tracker.stop()
                            else:
                                # Create a new event loop if needed
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(self.gpu_tracker.stop())
                                loop.close()
                        else:
                            # Synchronous implementation
                            self.gpu_tracker.stop()
                    self.logger.debug("GPU tracker stopped")
                except Exception as e:
                    self.logger.warning(f"Error stopping GPU tracker: {str(e)}")

            # Clear TensorFlow session if needed (TF1 compatibility)
            if not TF2 and keras is not None and hasattr(keras.backend, 'clear_session'):
                try:
                    keras.backend.clear_session()
                    self.logger.debug("Keras backend session cleared")
                except Exception as e:
                    self.logger.warning(f"Error clearing Keras session: {str(e)}")

            # TF2 cleanup - release GPU memory if possible
            if TF2 and hasattr(tf, 'keras') and hasattr(tf.keras.backend, 'clear_session'):
                try:
                    tf.keras.backend.clear_session()
                    self.logger.debug("TensorFlow Keras backend session cleared")
                except Exception as e:
                    self.logger.warning(f"Error clearing TF2 session: {str(e)}")

            # Reset internal state
            self._is_profiling = False
            self._profile_start_time = None

            self.logger.info("TensorFlow adapter resources cleaned up")

        except Exception as e:
            # Log but don't re-raise to ensure cleanup completes without crashing
            self.logger.error(f"Error during TensorFlow adapter cleanup: {str(e)}", exc_info=True)