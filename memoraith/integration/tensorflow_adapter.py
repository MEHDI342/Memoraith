"""
TensorFlow integration adapter for Memoraith profiler.
Handles profiling of TensorFlow models with comprehensive metrics collection.
"""

import tensorflow as tf
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional

from ..exceptions import MemoraithError
from .framework_adapter import FrameworkAdapter
from ..data_collection import TimeTracker, CPUMemoryTracker, GPUMemoryTracker
from ..config import config

# Properly handle TensorFlow import structure changes between versions
try:
    # For TensorFlow 2.x
    from tensorflow import keras
except ImportError:
    # Fallback for older versions
    try:
        import keras
    except ImportError:
        keras = None

class TensorFlowAdapter(FrameworkAdapter):
    """Advanced adapter for integrating with TensorFlow models."""

    def __init__(self, model):
        """
        Initialize the TensorFlow adapter.

        Args:
            model: The TensorFlow model to profile
        """
        super().__init__(model)
        self.time_tracker = TimeTracker()
        self.cpu_tracker = CPUMemoryTracker()
        self.gpu_tracker = GPUMemoryTracker() if tf.test.is_built_with_cuda() and config.enable_gpu else None
        self.original_call = None
        self.logger = logging.getLogger(__name__)
        self._original_fit = None
        self._original_predict = None
        self._original_evaluate = None

    def log_profiling_start(self) -> None:
        """Log when profiling starts."""
        self.logger.info("TensorFlow profiling started")

    def log_profiling_stop(self) -> None:
        """Log when profiling stops."""
        self.logger.info("TensorFlow profiling stopped")

    async def start_profiling(self) -> None:
        """Start profiling by wrapping model methods."""
        try:
            # Cache original methods
            if hasattr(self.model, 'call'):
                self.original_call = self.model.call
                self.model.call = self._wrapped_call

            # Wrap other important methods when available
            if hasattr(self.model, 'fit'):
                self._original_fit = self.model.fit
                self.model.fit = self._wrapped_fit

            if hasattr(self.model, 'predict'):
                self._original_predict = self.model.predict
                self.model.predict = self._wrapped_predict

            if hasattr(self.model, 'evaluate'):
                self._original_evaluate = self.model.evaluate
                self.model.evaluate = self._wrapped_evaluate

            # Start resource monitoring
            await self.cpu_tracker.start()
            if self.gpu_tracker:
                await self.gpu_tracker.start()

            # Log initial memory usage
            initial_memory = await self._get_initial_memory()
            self.data['initial_memory'] = initial_memory

            self.log_profiling_start()
            self.logger.debug(f"Initial memory usage: CPU={initial_memory.get('cpu_memory', 0):.2f}MB, GPU={initial_memory.get('gpu_memory', 0):.2f}MB")

        except Exception as e:
            self.logger.error(f"Error starting TensorFlow profiling: {str(e)}")
            await self.cleanup()
            raise MemoraithError(f"Failed to start TensorFlow profiling: {str(e)}")

    async def stop_profiling(self) -> None:
        """Stop profiling and restore original methods."""
        try:
            # Restore original methods
            if self.original_call:
                self.model.call = self.original_call
                self.original_call = None

            if self._original_fit:
                self.model.fit = self._original_fit
                self._original_fit = None

            if self._original_predict:
                self.model.predict = self._original_predict
                self._original_predict = None

            if self._original_evaluate:
                self.model.evaluate = self._original_evaluate
                self._original_evaluate = None

            # Stop resource monitoring
            await self.cpu_tracker.stop()
            if self.gpu_tracker:
                await self.gpu_tracker.stop()

            # Get final memory usage
            final_memory = await self._get_final_memory()
            self.data['final_memory'] = final_memory

            # Calculate memory delta
            if 'initial_memory' in self.data:
                self.data['memory_delta'] = {
                    'cpu_memory': final_memory.get('cpu_memory', 0) - self.data['initial_memory'].get('cpu_memory', 0),
                    'gpu_memory': final_memory.get('gpu_memory', 0) - self.data['initial_memory'].get('gpu_memory', 0)
                }

            self.log_profiling_stop()

        except Exception as e:
            self.logger.error(f"Error stopping TensorFlow profiling: {str(e)}")
            raise

    def _wrapped_call(self, *args, **kwargs):
        """
        Wrapped call method for profiling model inference.
        Captures timing and memory metrics for each forward pass.

        Returns:
            The result of the original call method
        """
        try:
            # Start metrics collection
            self.time_tracker.start('model_call')

            # Forward pass
            result = self.original_call(*args, **kwargs)

            # Stop timing
            self.time_tracker.stop('model_call')

            # Add metrics to data collection
            key = f"inference_{len([k for k in self.data.keys() if k.startswith('inference_')])}"
            self.data[key] = {
                'time': self.time_tracker.get_duration('model_call'),
                'timestamp': time.time()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error in wrapped call: {str(e)}")
            raise

    def _wrapped_fit(self, *args, **kwargs):
        """
        Wrapped fit method for profiling model training.

        Returns:
            The result of the original fit method
        """
        self.time_tracker.start('model_fit')

        # Apply callback to monitor each epoch
        callbacks = kwargs.get('callbacks', [])
        monitoring_callback = self._create_monitoring_callback()
        callbacks.append(monitoring_callback)
        kwargs['callbacks'] = callbacks

        # Call original fit method
        result = self._original_fit(*args, **kwargs)

        self.time_tracker.stop('model_fit')
        self.data['training'] = {
            'total_time': self.time_tracker.get_duration('model_fit'),
            'epochs': result.history.get('epochs', len(result.history.get('loss', []))),
            'history': {k: [float(v) for v in vals] for k, vals in result.history.items()}
        }

        return result

    def _wrapped_predict(self, *args, **kwargs):
        """
        Wrapped predict method for profiling inference.

        Returns:
            The result of the original predict method
        """
        self.time_tracker.start('model_predict')

        result = self._original_predict(*args, **kwargs)

        self.time_tracker.stop('model_predict')
        self.data['prediction'] = {
            'time': self.time_tracker.get_duration('model_predict'),
            'batch_size': len(args[0]) if args and hasattr(args[0], '__len__') else None,
            'timestamp': time.time()
        }

        return result

    def _wrapped_evaluate(self, *args, **kwargs):
        """
        Wrapped evaluate method for profiling evaluation.

        Returns:
            The result of the original evaluate method
        """
        self.time_tracker.start('model_evaluate')

        result = self._original_evaluate(*args, **kwargs)

        self.time_tracker.stop('model_evaluate')

        # Format metrics based on result type
        if isinstance(result, list):
            metrics = {}
            # Get metric names from model.metrics_names if available
            if hasattr(self.model, 'metrics_names'):
                for i, name in enumerate(self.model.metrics_names):
                    if i < len(result):
                        metrics[name] = float(result[i])
            else:
                # Fallback if metrics_names not available
                for i, value in enumerate(result):
                    metrics[f'metric_{i}'] = float(value)
        else:
            # Single metric result
            metrics = {'loss': float(result)}

        self.data['evaluation'] = {
            'time': self.time_tracker.get_duration('model_evaluate'),
            'metrics': metrics,
            'timestamp': time.time()
        }

        return result

    def _create_monitoring_callback(self):
        """
        Create a TensorFlow callback to monitor training progress.

        Returns:
            tf.keras.callbacks.Callback: A custom callback
        """
        class ProfilingCallback(keras.callbacks.Callback):
            def __init__(self, adapter):
                super().__init__()
                self.adapter = adapter
                self.epoch_times = []
                self.batch_times = []
                self.current_epoch_start = None
                self.current_batch_start = None

            def on_epoch_begin(self, epoch, logs=None):
                self.current_epoch_start = time.time()
                self.adapter.logger.debug(f"Starting epoch {epoch+1}")

            def on_epoch_end(self, epoch, logs=None):
                if self.current_epoch_start:
                    epoch_time = time.time() - self.current_epoch_start
                    self.epoch_times.append(epoch_time)

                    # Record metrics for this epoch
                    self.adapter.data[f'epoch_{epoch}'] = {
                        'time': epoch_time,
                        'metrics': {k: float(v) for k, v in (logs or {}).items()}
                    }

                    self.adapter.logger.debug(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

            def on_batch_begin(self, batch, logs=None):
                self.current_batch_start = time.time()

            def on_batch_end(self, batch, logs=None):
                if self.current_batch_start:
                    batch_time = time.time() - self.current_batch_start
                    self.batch_times.append(batch_time)

        return ProfilingCallback(self)

    async def _get_initial_memory(self) -> Dict[str, float]:
        """
        Get initial memory usage before profiling.

        Returns:
            Dict[str, float]: Memory usage statistics
        """
        cpu_memory = self.cpu_tracker.get_peak_memory() if hasattr(self.cpu_tracker, 'get_peak_memory') else {}
        gpu_memory = await self.gpu_tracker.get_peak_memory() if self.gpu_tracker else 0

        return {
            'cpu_memory': cpu_memory.get('rss', 0) if isinstance(cpu_memory, dict) else 0,
            'gpu_memory': gpu_memory
        }

    async def _get_final_memory(self) -> Dict[str, float]:
        """
        Get final memory usage after profiling.

        Returns:
            Dict[str, float]: Memory usage statistics
        """
        cpu_memory = self.cpu_tracker.get_peak_memory() if hasattr(self.cpu_tracker, 'get_peak_memory') else {}
        gpu_memory = await self.gpu_tracker.get_peak_memory() if self.gpu_tracker else 0

        return {
            'cpu_memory': cpu_memory.get('rss', 0) if isinstance(cpu_memory, dict) else 0,
            'gpu_memory': gpu_memory
        }

    async def profile_inference(self, input_data: Any) -> Dict[str, Any]:
        """
        Profile the inference process for a single input.

        Args:
            input_data: Input data for model inference

        Returns:
            Dict[str, Any]: Profiling metrics for the inference operation
        """
        if not self._is_profiling:
            await self.start_profiling()
            local_profiling = True
        else:
            local_profiling = False

        try:
            self.time_tracker.start('inference')
            output = self.model(input_data)
            self.time_tracker.stop('inference')

            inference_time = self.time_tracker.get_duration('inference')

            # Get current memory usage
            cpu_memory = self.cpu_tracker.get_peak_memory() if hasattr(self.cpu_tracker, 'get_peak_memory') else {}
            gpu_memory = await self.gpu_tracker.get_current_memory() if self.gpu_tracker else 0

            profiling_data = {
                'inference_time': inference_time,
                'cpu_memory': cpu_memory.get('rss', 0) if isinstance(cpu_memory, dict) else 0,
                'gpu_memory': gpu_memory,
                'input_shape': self._get_shape(input_data),
                'output_shape': self._get_shape(output),
                'timestamp': time.time()
            }

            # Store in the instance data
            key = f"inference_{len([k for k in self.data.keys() if k.startswith('inference_')])}"
            self.data[key] = profiling_data

            return profiling_data

        finally:
            if local_profiling:
                await self.stop_profiling()

    def _get_shape(self, tensor_data):
        """Get shape of tensor data safely."""
        try:
            if hasattr(tensor_data, 'shape'):
                return list(tensor_data.shape)
            elif isinstance(tensor_data, (list, tuple)) and tensor_data and hasattr(tensor_data[0], 'shape'):
                return [list(t.shape) for t in tensor_data]
            return None
        except:
            return None

    async def profile_training_step(self, input_data: Any, target: Any) -> Dict[str, Any]:
        """
        Profile a single training step.

        Args:
            input_data: Input data for training
            target: Target/label data for training

        Returns:
            Dict[str, Any]: Profiling metrics for the training step
        """
        if not self._is_profiling:
            await self.start_profiling()
            local_profiling = True
        else:
            local_profiling = False

        try:
            # Verify model is compiled
            if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                raise MemoraithError("Model must be compiled before training")

            # Use model's training step if available (for custom models), otherwise use GradientTape
            self.time_tracker.start('training_step')

            if hasattr(self.model, 'train_step'):
                # Custom training step method is available
                packed_data = self._pack_data(input_data, target)
                result = self.model.train_step(packed_data)
            else:
                # Use manual training step with GradientTape
                with tf.GradientTape() as tape:
                    predictions = self.model(input_data, training=True)
                    loss = self.model.compiled_loss(target, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                # Compute metrics
                self.model.compiled_metrics.update_state(target, predictions)
                result = {m.name: m.result().numpy() for m in self.model.metrics}
                result['loss'] = float(loss.numpy())

            self.time_tracker.stop('training_step')

            # Get memory info
            cpu_memory = self.cpu_tracker.get_peak_memory() if hasattr(self.cpu_tracker, 'get_peak_memory') else {}
            gpu_memory = await self.gpu_tracker.get_current_memory() if self.gpu_tracker else 0

            training_data = {
                'step_time': self.time_tracker.get_duration('training_step'),
                'loss': float(result['loss']) if 'loss' in result else None,
                'metrics': {k: float(v) if hasattr(v, 'numpy') else float(v) for k, v in result.items() if k != 'loss'},
                'cpu_memory': cpu_memory.get('rss', 0) if isinstance(cpu_memory, dict) else 0,
                'gpu_memory': gpu_memory,
                'timestamp': time.time()
            }

            # Store in instance data
            key = f"train_step_{len([k for k in self.data.keys() if k.startswith('train_step_')])}"
            self.data[key] = training_data

            return training_data

        finally:
            if local_profiling:
                await self.stop_profiling()

    def _pack_data(self, x, y):
        """Pack input and target data as expected by model.train_step."""
        if isinstance(x, dict) and y is None:
            return x  # Data already in expected format
        return (x, y)

    async def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the model architecture.

        Returns:
            Dict[str, Any]: Model architecture summary
        """
        summary = {}

        try:
            # Get basic model info
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

            # Get layer information
            if hasattr(self.model, 'layers'):
                layers_info = []
                for layer in self.model.layers:
                    layer_info = {
                        'name': layer.name,
                        'type': layer.__class__.__name__,
                        'params': layer.count_params(),
                        'trainable': layer.trainable,
                        'input_shape': list(layer.input_shape) if hasattr(layer, 'input_shape') else None,
                        'output_shape': list(layer.output_shape) if hasattr(layer, 'output_shape') else None,
                    }
                    layers_info.append(layer_info)
                summary['layers'] = layers_info

            # Get optimizer info
            if hasattr(self.model, 'optimizer') and self.model.optimizer:
                optimizer = self.model.optimizer
                summary['optimizer'] = {
                    'name': optimizer.__class__.__name__,
                    'learning_rate': float(keras.backend.get_value(optimizer.lr)) if hasattr(optimizer, 'lr') else None,
                    'parameters': {k: v for k, v in optimizer.get_config().items() if k != 'name'}
                }

            # Get loss function info
            if hasattr(self.model, 'loss') and self.model.loss:
                loss = self.model.loss
                if callable(loss):
                    summary['loss'] = {'name': loss.__name__}
                elif isinstance(loss, str):
                    summary['loss'] = {'name': loss}
                else:
                    summary['loss'] = {'name': loss.__class__.__name__}

            # Calculate model size estimation
            if hasattr(self.model, 'weights'):
                size_bytes = sum(keras.backend.count_params(w) * 4 for w in self.model.weights)  # Assuming float32
                summary['model_size_mb'] = size_bytes / (1024 * 1024)

            return summary

        except Exception as e:
            self.logger.error(f"Error generating model summary: {str(e)}")
            return {'error': str(e)}

    async def cleanup(self) -> None:
        """
        Clean up resources used during profiling.
        Restores original model methods and clears session if needed.
        """
        try:
            # Restore original methods if not already done
            if self.original_call:
                self.model.call = self.original_call
                self.original_call = None

            if self._original_fit:
                self.model.fit = self._original_fit
                self._original_fit = None

            if self._original_predict:
                self.model.predict = self._original_predict
                self._original_predict = None

            if self._original_evaluate:
                self.model.evaluate = self._original_evaluate
                self._original_evaluate = None

            # Clear TensorFlow session if needed
            if keras and hasattr(keras.backend, 'clear_session'):
                keras.backend.clear_session()

            self.logger.info("TensorFlow adapter resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    async def validate_model(self) -> bool:
        """
        Validate the TensorFlow model.

        Returns:
            bool: True if the model is valid, False otherwise
        """
        try:
            # Check if it's a proper TensorFlow model
            if not hasattr(self.model, 'call'):
                self.logger.error("Model does not have a 'call' method")
                return False

            # Verify model is built/has weights
            if not hasattr(self.model, 'weights') or not self.model.weights:
                self.logger.warning("Model does not have weights (not built)")

            # Check if the model has been compiled
            if hasattr(self.model, 'compiled_loss') and self.model.compiled_loss is None:
                self.logger.warning("Model has not been compiled")

            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return False