import tensorflow as tf
import time
from typing import Dict, Any, List
import logging
from .framework_adapter import FrameworkAdapter
from ..data_collection import TimeTracker, CPUMemoryTracker, GPUMemoryTracker
from ..config import config

class TensorFlowAdapter(FrameworkAdapter):
    """Advanced adapter for integrating with TensorFlow models."""

    def __init__(self, model: tf.keras.Model):
        super().__init__(model)
        self.time_tracker = TimeTracker()
        self.cpu_tracker = CPUMemoryTracker()
        self.gpu_tracker = GPUMemoryTracker() if tf.test.is_built_with_cuda() and config.enable_gpu else None
        self.original_call = None
        self.logger = logging.getLogger(__name__)

    async def start_profiling(self) -> None:
        """Start profiling by wrapping the model's call method."""
        self.original_call = self.model.call
        self.model.call = self._wrapped_call

        await self.cpu_tracker.start()
        if self.gpu_tracker:
            await self.gpu_tracker.start()
        await self.log_profiling_start()

    async def stop_profiling(self) -> None:
        """Restore the original call method after profiling."""
        if self.original_call:
            self.model.call = self.original_call
            self.original_call = None

        await self.cpu_tracker.stop()
        if self.gpu_tracker:
            await self.gpu_tracker.stop()
        await self.log_profiling_stop()

    async def _wrapped_call(self, *args, **kwargs):
        """Wrapped call method for profiling each layer."""
        for layer in self.model.layers:
            layer_name = f"{layer.__class__.__name__}_{id(layer)}"
            self.time_tracker.start(layer_name)
            output = layer(*args, **kwargs)
            self.time_tracker.stop(layer_name)

            try:
                self.data[layer_name] = self.data.get(layer_name, {})
                self.data[layer_name]['time'] = self.time_tracker.get_duration(layer_name)
                self.data[layer_name]['parameters'] = layer.count_params()

                if config.enable_memory:
                    self.data[layer_name]['cpu_memory'] = await self.cpu_tracker.get_peak_memory()
                    if self.gpu_tracker:
                        self.data[layer_name]['gpu_memory'] = await self.gpu_tracker.get_peak_memory()

                if hasattr(output, 'shape'):
                    self.data[layer_name]['output_shape'] = output.shape.as_list()
            except Exception as e:
                self.logger.error(f"Error in _wrapped_call for layer {layer_name}: {str(e)}")

            args = (output,)

        return await self.original_call(*args, **kwargs)

    async def profile_inference(self, input_data: tf.Tensor) -> Dict[str, Any]:
        """Profile the inference process for a single input."""
        await self.start_profiling()
        try:
            start_time = time.perf_counter()
            output = self.model(input_data)
            end_time = time.perf_counter()

            inference_time = end_time - start_time
            profiling_data = self.data.copy()
            profiling_data['inference_time'] = inference_time

            return profiling_data
        finally:
            await self.stop_profiling()

    async def profile_training_step(self, input_data: tf.Tensor, target: tf.Tensor) -> Dict[str, Any]:
        """Profile a single training step."""
        await self.start_profiling()
        try:
            optimizer = self.model.optimizer
            loss_function = self.model.loss

            if optimizer is None or loss_function is None:
                raise ValueError("Optimizer or loss function not properly configured")

            start_time = time.perf_counter()
            with tf.GradientTape() as tape:
                predictions = self.model(input_data, training=True)
                loss = loss_function(target, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            end_time = time.perf_counter()

            step_time = end_time - start_time
            profiling_data = self.data.copy()
            profiling_data['step_time'] = step_time
            profiling_data['loss'] = loss.numpy().item()

            return profiling_data
        finally:
            await self.stop_profiling()

    async def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture."""
        summary = {}
        total_params = self.model.count_params()
        trainable_params = sum(tf.keras.backend.count_params(w) for w in self.model.trainable_weights)

        summary['total_params'] = total_params
        summary['trainable_params'] = trainable_params
        summary['non_trainable_params'] = total_params - trainable_params
        summary['model_size_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32

        return summary

    async def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific layer."""
        for layer in self.model.layers:
            if layer.name == layer_name:
                return {
                    'type': type(layer).__name__,
                    'parameters': layer.count_params(),
                    'trainable_parameters': sum(tf.keras.backend.count_params(w) for w in layer.trainable_weights),
                    'input_shape': layer.input_shape,
                    'output_shape': layer.output_shape,
                    'activation': layer.activation.__name__ if hasattr(layer, 'activation') else None,
                }
        return {}

    async def profile_memory_usage(self) -> Dict[str, float]:
        """Profile the memory usage of the model."""
        tf.keras.backend.clear_session()

        initial_mem = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.test.is_gpu_available() else 0

        # Profile memory usage during forward pass
        dummy_input = tf.random.normal(shape=[1] + self.model.input_shape[1:])
        self.model(dummy_input)
        forward_mem = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.test.is_gpu_available() else 0

        # Profile memory usage during backward pass
        with tf.GradientTape() as tape:
            predictions = self.model(dummy_input)
            loss = tf.reduce_mean(predictions)
        _ = tape.gradient(loss, self.model.trainable_variables)
        backward_mem = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.test.is_gpu_available() else 0

        return {
            'initial_memory': initial_mem / (1024 * 1024),  # Convert to MB
            'forward_pass_memory': (forward_mem - initial_mem) / (1024 * 1024),
            'backward_pass_memory': (backward_mem - forward_mem) / (1024 * 1024),
            'total_memory': (backward_mem - initial_mem) / (1024 * 1024)
        }

    def get_flops(self) -> int:
        """Calculate the total number of FLOPs for the model."""
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(tf.compat.v1.get_default_graph(), run_meta=run_meta, cmd='scope', options=opts)
        return flops.total_float_ops

    async def get_current_time(self) -> float:
        """Get the current time in a high-precision format."""
        return time.perf_counter()

    async def export_model(self, path: str, format: str) -> None:
        """Export the model to a specified format."""
        if format.lower() == 'savedmodel':
            tf.saved_model.save(self.model, path)
        elif format.lower() == 'h5':
            self.model.save(path, save_format='h5')
        else:
            raise ValueError(f"Unsupported export format: {format}")
        self.logger.info(f"Model exported to {format} format at {path}")

    async def visualize_model(self, output_path: str) -> None:
        """Generate a visual representation of the model architecture."""
        tf.keras.utils.plot_model(self.model, to_file=output_path, show_shapes=True, show_layer_names=True)
        self.logger.info(f"Model visualization saved to {output_path}")

    async def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the current optimizer."""
        optimizer = self.model.optimizer
        return {
            'name': optimizer.__class__.__name__,
            'learning_rate': float(tf.keras.backend.get_value(optimizer.lr)),
            'parameters': {k: v.numpy() for k, v in optimizer.get_config().items() if k != 'name'}
        }

    async def get_loss_function_info(self) -> Dict[str, Any]:
        """Get information about the current loss function."""
        loss = self.model.loss
        if callable(loss):
            return {'name': loss.__name__}
        elif isinstance(loss, str):
            return {'name': loss}
        elif isinstance(loss, tf.keras.losses.Loss):
            return {
                'name': loss.__class__.__name__,
                'parameters': loss.get_config()
            }
        else:
            return {'error': 'Unknown loss type'}
