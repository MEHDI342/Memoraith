import torch
import logging
from typing import Dict, Any, List, Optional
import asyncio
import time
from .framework_adapter import FrameworkAdapter
from ..data_collection import TimeTracker, CPUMemoryTracker, GPUMemoryTracker
from ..config import config

class PyTorchAdapter(FrameworkAdapter):
    """Complete adapter for integrating with PyTorch models."""

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)
        self.time_tracker = TimeTracker()
        self.cpu_tracker = CPUMemoryTracker()
        self.gpu_tracker = GPUMemoryTracker() if torch.cuda.is_available() and config.enable_gpu else None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.enable_gpu else "cpu")
        self.model.to(self.device)

    async def start_profiling(self) -> None:
        """Start profiling by registering hooks."""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                pre_handle = module.register_forward_pre_hook(self._pre_forward_hook)
                post_handle = module.register_forward_hook(self._forward_hook)
                self.handles.extend([pre_handle, post_handle])

        await self.cpu_tracker.start()
        if self.gpu_tracker:
            await self.gpu_tracker.start()

    async def stop_profiling(self) -> None:
        """Remove hooks after profiling."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        await self.cpu_tracker.stop()
        if self.gpu_tracker:
            await self.gpu_tracker.stop()

    async def _forward_hook(self, module: torch.nn.Module, input: Any, output: Any) -> None:
        layer_name = f"{module.__class__.__name__}_{id(module)}"
        self.time_tracker.stop(layer_name)

        try:
            self.data[layer_name] = self.data.get(layer_name, {})
            self.data[layer_name]['time'] = self.time_tracker.get_duration(layer_name)
            self.data[layer_name]['parameters'] = sum(p.numel() for p in module.parameters())

            if config.enable_memory:
                self.data[layer_name]['cpu_memory'] = await self.cpu_tracker.get_peak_memory()
                if self.gpu_tracker:
                    self.data[layer_name]['gpu_memory'] = await self.gpu_tracker.get_peak_memory()

            if isinstance(output, torch.Tensor):
                self.data[layer_name]['output_shape'] = list(output.shape)
            elif isinstance(output, (list, tuple)) and all(isinstance(o, torch.Tensor) for o in output):
                self.data[layer_name]['output_shape'] = [list(o.shape) for o in output]
        except Exception as e:
            self.logger.error(f"Error in forward hook for layer {layer_name}: {str(e)}")

    async def _pre_forward_hook(self, module: torch.nn.Module, input: Any) -> None:
        layer_name = f"{module.__class__.__name__}_{id(module)}"
        self.time_tracker.start(layer_name)

    async def profile_inference(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """Profile the inference process for a single input."""
        await self.start_profiling()

        try:
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                output = self.model(input_data.to(self.device))
                end_time.record()

                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

            profiling_data = self.data.copy()
            profiling_data['inference_time'] = inference_time

            return profiling_data
        finally:
            await self.stop_profiling()

    async def profile_training_step(self, input_data: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Profile a single training step."""
        await self.start_profiling()

        try:
            optimizer = config.get_optimizer(self.model.parameters())
            loss_function = config.get_loss_function()

            if optimizer is None or loss_function is None:
                raise ValueError("Optimizer or loss function not properly configured")

            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)

            start_time.record()
            optimizer.zero_grad()
            output = self.model(input_data.to(self.device))
            loss = loss_function(output, target.to(self.device))
            loss.backward()
            optimizer.step()
            end_time.record()

            torch.cuda.synchronize()
            step_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds

            profiling_data = self.data.copy()
            profiling_data['step_time'] = step_time
            profiling_data['loss'] = loss.item()

            return profiling_data
        finally:
            await self.stop_profiling()

    async def profile_full_training(self, train_loader: torch.utils.data.DataLoader, num_epochs: Optional[int] = None) -> List[Dict[str, Any]]:
        """Profile the full training process."""
        epochs_to_run = num_epochs or config.max_epochs
        epoch_data = []

        for epoch in range(epochs_to_run):
            epoch_start_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time = torch.cuda.Event(enable_timing=True)

            epoch_start_time.record()
            epoch_loss = 0.0
            batch_data = []

            for batch_idx, (data, target) in enumerate(train_loader):
                batch_profile = await self.profile_training_step(data, target)
                epoch_loss += batch_profile['loss']
                batch_data.append(batch_profile)

            epoch_end_time.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start_time.elapsed_time(epoch_end_time) / 1000  # Convert to seconds

            epoch_data.append({
                'epoch': epoch,
                'epoch_time': epoch_time,
                'epoch_loss': epoch_loss / len(train_loader),
                'batch_data': batch_data
            })

        return epoch_data

    async def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model architecture."""
        summary = {}
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        summary['total_params'] = total_params
        summary['trainable_params'] = trainable_params
        summary['non_trainable_params'] = total_params - trainable_params
        summary['model_size_mb'] = total_params * 4 / (1024 * 1024)  # Assuming float32

        return summary

    async def get_layer_info(self, layer_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific layer."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return {
                    'type': type(module).__name__,
                    'parameters': sum(p.numel() for p in module.parameters()),
                    'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad),
                    'input_shape': getattr(module, 'in_features', None) or getattr(module, 'in_channels', None),
                    'output_shape': getattr(module, 'out_features', None) or getattr(module, 'out_channels', None),
                    'activation': getattr(module, 'activation', None),
                    'dropout_rate': getattr(module, 'p', None) if isinstance(module, torch.nn.Dropout) else None,
                }
        return {}

    async def profile_memory_usage(self) -> Dict[str, float]:
        """Profile the memory usage of the model."""
        torch.cuda.empty_cache()

        def get_memory_usage():
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
            else:
                import psutil
                return psutil.Process().memory_info().rss / 1024**2  # Convert to MB

        initial_mem = get_memory_usage()

        # Profile memory usage during forward pass
        dummy_input = torch.randn(1, *self.model.input_size).to(self.device)
        self.model(dummy_input)
        forward_mem = get_memory_usage()

        # Profile memory usage during backward pass
        loss = self.model(dummy_input).sum()
        loss.backward()
        backward_mem = get_memory_usage()

        return {
            'initial_memory': initial_mem,
            'forward_pass_memory': forward_mem - initial_mem,
            'backward_pass_memory': backward_mem - forward_mem,
            'total_memory': backward_mem - initial_mem
        }

    def get_flops(self) -> int:
        """Calculate the total number of FLOPs for the model."""
        from torch.autograd import Variable

        flops = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                flops += (2 * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] - 1) * (module.input_size[2] * module.input_size[3] / (module.stride[0] * module.stride[1]))
            elif isinstance(module, torch.nn.Linear):
                flops += 2 * module.in_features * module.out_features - 1

        return int(flops)

    async def visualize_model(self, output_path: str) -> None:
        """Generate a visual representation of the model architecture."""
        from torchviz import make_dot

        dummy_input = torch.randn(1, *self.model.input_size).to(self.device)
        y = self.model(dummy_input)

        dot = make_dot(y, params=dict(self.model.named_parameters()))
        dot.render(output_path, format='png')
        self.logger.info(f"Model visualization saved to {output_path}.png")

    async def export_onnx(self, output_path: str) -> None:
        """Export the model to ONNX format."""
        dummy_input = torch.randn(1, *self.model.input_size).to(self.device)
        torch.onnx.export(self.model, dummy_input, output_path, verbose=True)
        self.logger.info(f"Model exported to ONNX format at {output_path}")

    async def export_model(self, path: str, format: str) -> None:
        """Export the model to a specified format."""
        if format.lower() == 'onnx':
            await self.export_onnx(path)
        elif format.lower() == 'torchscript':
            script_module = torch.jit.script(self.model)
            torch.jit.save(script_module, path)
            self.logger.info(f"Model exported to TorchScript format at {path}")
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def get_current_time(self) -> float:
        """Get the current time in a high-precision format."""
        return time.perf_counter()

    async def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the current optimizer."""
        optimizer = getattr(self.model, 'optimizer', None)
        if optimizer:
            return {
                'name': optimizer.__class__.__name__,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'parameters': {k: v for k, v in optimizer.defaults.items() if k != 'params'}
            }
        return {'error': 'No optimizer found'}

    async def get_loss_function_info(self) -> Dict[str, Any]:
        """Get information about the current loss function."""
        loss_fn = getattr(self.model, 'loss_fn', None)
        if loss_fn:
            return {
                'name': loss_fn.__class__.__name__,
                'parameters': {k: v for k, v in loss_fn.__dict__.items() if not k.startswith('_')}
            }
        return {'error': 'No loss function found'}

    def __del__(self):
        """Cleanup method to ensure all profiling is stopped."""
        asyncio.run(self.stop_profiling())