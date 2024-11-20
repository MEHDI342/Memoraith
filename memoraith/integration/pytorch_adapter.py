import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.cuda.nvtx as nvtx
import psutil
import threading
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
import queue
from collections import defaultdict

class ProfilingLevel(Enum):
    BASIC = "basic"          # Basic metrics only
    MEMORY = "memory"        # Memory-focused profiling
    COMPUTE = "compute"      # Computation-focused profiling
    FULL = "full"           # All metrics and features

@dataclass
class LayerProfile:
    """Detailed layer profiling information"""
    name: str
    layer_type: str
    input_shape: Tuple
    output_shape: Tuple
    parameters: int
    cpu_memory: float
    gpu_memory: float
    compute_time: float
    flops: int
    backward_time: float
    gradient_norm: float
    activation_memory: float
    buffer_memory: float
    cuda_memory_allocated: float
    cuda_memory_cached: float
    cuda_utilization: float
    peak_memory: float

class PyTorchAdapter:
    """Advanced PyTorch model profiling adapter"""

    def __init__(
            self,
            model: nn.Module,
            level: ProfilingLevel = ProfilingLevel.FULL,
            log_dir: str = "profiling_logs",
            device: Optional[torch.device] = None
    ):
        self.model = model
        self.level = level
        self.log_dir = Path(log_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Profiling data structures
        self.layer_profiles: Dict[str, LayerProfile] = {}
        self.training_history: List[Dict[str, float]] = []
        self.memory_traces: Dict[str, List[float]] = defaultdict(list)
        self.gradient_history: Dict[str, List[float]] = defaultdict(list)
        self.activation_maps: Dict[str, List[torch.Tensor]] = {}
        self.bottlenecks: Set[str] = set()

        # Performance monitoring
        self._monitoring_queue = queue.Queue()
        self._stop_monitoring = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None

        # CUDA events for precise timing
        self.cuda_events: Dict[str, torch.cuda.Event] = {}

        # Move model to device
        self.model.to(self.device)
        self._attach_hooks()
        self.logger.info(f"PyTorch adapter initialized with profiling level: {level.value}")

    def _setup_logging(self) -> None:
        """Configure detailed logging"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.log_dir / "pytorch_profiling.log")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG)

    def _attach_hooks(self) -> None:
        """Attach comprehensive profiling hooks to model"""
        for name, module in self.model.named_modules():
            # Forward pre-hook for input analysis
            module.register_forward_pre_hook(self._forward_pre_hook(name))

            # Forward hook for output analysis
            module.register_forward_hook(self._forward_hook(name))

            # Backward hook for gradient analysis
            if hasattr(module, 'weight') and module.weight is not None:
                module.register_full_backward_hook(self._backward_hook(name))

            self.cuda_events[f"{name}_forward_start"] = torch.cuda.Event(enable_timing=True)
            self.cuda_events[f"{name}_forward_end"] = torch.cuda.Event(enable_timing=True)
            self.cuda_events[f"{name}_backward_start"] = torch.cuda.Event(enable_timing=True)
            self.cuda_events[f"{name}_backward_end"] = torch.cuda.Event(enable_timing=True)

    def _forward_pre_hook(self, name: str):
        """Pre-forward pass hook for input analysis"""
        def hook(module: nn.Module, input: Tuple[torch.Tensor]):
            if self.level != ProfilingLevel.BASIC:
                try:
                    # Start CUDA timing
                    event = self.cuda_events[f"{name}_forward_start"]
                    event.record()

                    # Record input statistics
                    if input[0].requires_grad:
                        input_size = input[0].element_size() * input[0].nelement()
                        self.memory_traces[f"{name}_input"].append(input_size)

                    # CUDA memory tracking
                    if torch.cuda.is_available():
                        self._record_cuda_memory(name, "pre_forward")

                except Exception as e:
                    self.logger.error(f"Error in forward pre-hook for {name}: {str(e)}")
        return hook

    def _forward_hook(self, name: str):
        """Post-forward pass hook for comprehensive analysis"""
        def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
            try:
                # End CUDA timing
                event_start = self.cuda_events[f"{name}_forward_start"]
                event_end = self.cuda_events[f"{name}_forward_end"]
                event_end.record()

                # Basic metrics
                profile = LayerProfile(
                    name=name,
                    layer_type=module.__class__.__name__,
                    input_shape=tuple(input[0].shape),
                    output_shape=tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                    parameters=sum(p.numel() for p in module.parameters() if p.requires_grad),
                    cpu_memory=0,
                    gpu_memory=0,
                    compute_time=0,
                    flops=self._calculate_flops(module, input[0], output),
                    backward_time=0,
                    gradient_norm=0,
                    activation_memory=0,
                    buffer_memory=0,
                    cuda_memory_allocated=0,
                    cuda_memory_cached=0,
                    cuda_utilization=0,
                    peak_memory=0
                )

                if self.level in (ProfilingLevel.MEMORY, ProfilingLevel.FULL):
                    # Memory analysis
                    self._analyze_memory(name, module, output, profile)

                if self.level in (ProfilingLevel.COMPUTE, ProfilingLevel.FULL):
                    # Computation analysis
                    self._analyze_computation(name, module, profile, event_start, event_end)

                # Store activation maps for visualization
                if isinstance(output, torch.Tensor):
                    self.activation_maps[name] = output.detach().cpu().numpy()

                # Update profile
                self.layer_profiles[name] = profile

                # Check for bottlenecks
                self._check_bottlenecks(name, profile)

            except Exception as e:
                self.logger.error(f"Error in forward hook for {name}: {str(e)}")
        return hook

    def _backward_hook(self, name: str):
        """Backward pass hook for gradient analysis"""
        def hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            try:
                # Record gradient timing
                event_start = self.cuda_events[f"{name}_backward_start"]
                event_end = self.cuda_events[f"{name}_backward_end"]
                event_start.record()

                if self.level in (ProfilingLevel.COMPUTE, ProfilingLevel.FULL):
                    # Gradient analysis
                    if grad_output[0] is not None:
                        grad_norm = grad_output[0].norm().item()
                        self.gradient_history[name].append(grad_norm)

                        if name in self.layer_profiles:
                            self.layer_profiles[name].gradient_norm = grad_norm

                event_end.record()

            except Exception as e:
                self.logger.error(f"Error in backward hook for {name}: {str(e)}")
        return hook

    def _analyze_memory(self, name: str, module: nn.Module, output: torch.Tensor, profile: LayerProfile) -> None:
        """Detailed memory analysis for a layer"""
        try:
            # CPU memory
            profile.cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # GPU memory if available
            if torch.cuda.is_available():
                profile.gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                profile.cuda_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024
                profile.cuda_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024

                # Peak memory tracking
                profile.peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            # Activation memory
            if isinstance(output, torch.Tensor):
                profile.activation_memory = output.element_size() * output.nelement() / 1024 / 1024

            # Buffer memory
            profile.buffer_memory = sum(b.element_size() * b.nelement() for b in module.buffers()) / 1024 / 1024

        except Exception as e:
            self.logger.error(f"Error in memory analysis for {name}: {str(e)}")

    def _analyze_computation(self, name: str, module: nn.Module, profile: LayerProfile,
                             event_start: torch.cuda.Event, event_end: torch.cuda.Event) -> None:
        """Detailed computation analysis for a layer"""
        try:
            # CUDA timing
            torch.cuda.synchronize()
            profile.compute_time = event_start.elapsed_time(event_end) / 1000  # Convert to seconds

            # CUDA utilization
            if torch.cuda.is_available():
                profile.cuda_utilization = torch.cuda.utilization()

        except Exception as e:
            self.logger.error(f"Error in computation analysis for {name}: {str(e)}")

    def _calculate_flops(self, module: nn.Module, input_tensor: torch.Tensor,
                         output_tensor: torch.Tensor) -> int:
        """Calculate FLOPs for different layer types"""
        try:
            if isinstance(module, nn.Conv2d):
                return self._conv2d_flops(module, input_tensor)
            elif isinstance(module, nn.Linear):
                return self._linear_flops(module)
            elif isinstance(module, nn.LSTM):
                return self._lstm_flops(module)
            return 0
        except Exception as e:
            self.logger.error(f"Error calculating FLOPs: {str(e)}")
            return 0

    def _conv2d_flops(self, module: nn.Conv2d, input_tensor: torch.Tensor) -> int:
        """Calculate FLOPs for Conv2d layer"""
        batch_size = input_tensor.shape[0]
        output_height = input_tensor.shape[2] // module.stride[0]
        output_width = input_tensor.shape[3] // module.stride[1]

        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
        flops_per_instance = kernel_ops * module.out_channels
        total_flops = flops_per_instance * output_height * output_width * batch_size

        return total_flops

    def _check_bottlenecks(self, name: str, profile: LayerProfile) -> None:
        """Identify performance bottlenecks"""
        try:
            # Memory bottleneck
            if profile.gpu_memory > 1000:  # More than 1GB
                self.bottlenecks.add(f"{name}_memory")

            # Compute bottleneck
            if profile.compute_time > 0.1:  # More than 100ms
                self.bottlenecks.add(f"{name}_compute")

            # Gradient bottleneck
            if profile.gradient_norm > 100:
                self.bottlenecks.add(f"{name}_gradient")

        except Exception as e:
            self.logger.error(f"Error checking bottlenecks for {name}: {str(e)}")

    def start_monitoring(self) -> None:
        """Start continuous resource monitoring"""
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_resources)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()

    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join()

    def _monitor_resources(self) -> None:
        """Continuous resource monitoring thread"""
        while not self._stop_monitoring.is_set():
            try:
                gpu_info = {
                    'memory_allocated': torch.cuda.memory_allocated() / 1024 / 1024,
                    'memory_cached': torch.cuda.memory_reserved() / 1024 / 1024,
                    'utilization': torch.cuda.utilization()
                } if torch.cuda.is_available() else {}

                cpu_info = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'swap_percent': psutil.swap_memory().percent
                }

                self._monitoring_queue.put({
                    'timestamp': time.time(),
                    'gpu': gpu_info,
                    'cpu': cpu_info
                })

                time.sleep(0.1)  # 100ms interval

            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {str(e)}")

    def get_profiling_results(self) -> Dict[str, Any]:
        """Get comprehensive profiling results"""
        return {
            'layer_profiles': {name: vars(profile) for name, profile in self.layer_profiles.items()},
            'bottlenecks': list(self.bottlenecks),
            'training_history': self.training_history,
            'gradient_history': dict(self.gradient_history),
            'memory_traces': dict(self.memory_traces),
            'monitoring_data': self._get_monitoring_data()
        }


    def _get_monitoring_data(self) -> List[Dict[str, Any]]:
        """Get collected monitoring data"""
        data = []
        while not self._monitoring_queue.empty():
            data.append(self._monitoring_queue.get())
        return data


    def save_results(self, filename: str) -> None:
        """Save profiling results to file"""
        try:
            results = self.get_profiling_results()

            # Convert numpy arrays to lists for JSON serialization
            for name, activation in self.activation_maps.items():
                if isinstance(activation, np.ndarray):
                    results[f'activation_{name}'] = activation.tolist()

            # Save to file
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)

            self.logger.info(f"Profiling results saved to {filename}")

        except Exception as e:
            self.logger.error(f"Error saving profiling results: {str(e)}")
            raise

    async def profile_training_step(self,
                                    input_data: torch.Tensor,
                                    target: torch.Tensor,
                                    optimizer: torch.optim.Optimizer,
                                    criterion: nn.Module) -> Dict[str, float]:
        """Profile a single training step with detailed metrics"""
        try:
            # Start timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass timing
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)

            forward_start.record()
            output = self.model(input_data.to(self.device))
            forward_end.record()

            # Loss computation
            loss = criterion(output, target.to(self.device))

            # Backward pass timing
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)

            backward_start.record()
            loss.backward()
            backward_end.record()

            # Optimizer step timing
            optim_start = torch.cuda.Event(enable_timing=True)
            optim_end = torch.cuda.Event(enable_timing=True)

            optim_start.record()
            optimizer.step()
            optim_end.record()

            # Synchronize and record timings
            torch.cuda.synchronize()

            metrics = {
                'total_time': start_event.elapsed_time(end_event) / 1000,
                'forward_time': forward_start.elapsed_time(forward_end) / 1000,
                'backward_time': backward_start.elapsed_time(backward_end) / 1000,
                'optimizer_time': optim_start.elapsed_time(optim_end) / 1000,
                'loss': loss.item(),
                'gpu_memory': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
                'cpu_memory': psutil.Process().memory_info().rss / 1024 / 1024
            }

            # Record gradient norms
            grad_norms = {
                name: param.grad.norm().item()
                for name, param in self.model.named_parameters()
                if param.grad is not None
            }
            metrics['gradient_norms'] = grad_norms

            # Add to training history
            self.training_history.append(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error in training step profiling: {str(e)}")
            raise

    def analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture for optimization opportunities"""
        try:
            analysis = {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'layer_distribution': {},
                'memory_distribution': {},
                'compute_distribution': {},
                'optimization_suggestions': []
            }

            # Analyze layer distribution
            layer_types = {}
            for name, module in self.model.named_modules():
                layer_type = module.__class__.__name__
                layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
            analysis['layer_distribution'] = layer_types

            # Memory and compute analysis
            total_memory = sum(p.memory_usage for p in self.layer_profiles.values())
            total_compute = sum(p.compute_time for p in self.layer_profiles.values())

            for name, profile in self.layer_profiles.items():
                analysis['memory_distribution'][name] = profile.memory_usage / total_memory
                analysis['compute_distribution'][name] = profile.compute_time / total_compute

            # Generate optimization suggestions
            self._generate_optimization_suggestions(analysis)

            return analysis

        except Exception as e:
            self.logger.error(f"Error in model architecture analysis: {str(e)}")
            raise

    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> None:
        """Generate optimization suggestions based on profiling data"""
        suggestions = []

        # Memory optimization suggestions
        for name, memory_percent in analysis['memory_distribution'].items():
            if memory_percent > 0.2:  # Layer uses more than 20% of total memory
                suggestions.append({
                    'type': 'memory',
                    'layer': name,
                    'suggestion': 'Consider reducing layer size or using gradient checkpointing',
                    'impact': 'high'
                })

        # Compute optimization suggestions
        for name, compute_percent in analysis['compute_distribution'].items():
            if compute_percent > 0.3:  # Layer uses more than 30% of compute time
                suggestions.append({
                    'type': 'compute',
                    'layer': name,
                    'suggestion': 'Consider using a more efficient layer type or reducing complexity',
                    'impact': 'high'
                })

        # Model-wide suggestions
        if analysis['total_parameters'] > 1e8:  # More than 100M parameters
            suggestions.append({
                'type': 'model',
                'suggestion': 'Consider model pruning or quantization',
                'impact': 'medium'
            })

        analysis['optimization_suggestions'] = suggestions

    def visualize_profiling_results(self, output_dir: str) -> None:
        """Generate comprehensive visualization of profiling results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Memory usage over time
            plt.figure(figsize=(12, 6))
            plt.plot(self.training_history)
            plt.title('Memory Usage Over Time')
            plt.xlabel('Training Step')
            plt.ylabel('Memory (MB)')
            plt.savefig(output_path / 'memory_usage.png')
            plt.close()

            # Layer compute times
            compute_times = {name: profile.compute_time for name, profile in self.layer_profiles.items()}
            plt.figure(figsize=(15, 8))
            sns.barplot(x=list(compute_times.values()), y=list(compute_times.keys()))
            plt.title('Layer Compute Times')
            plt.xlabel('Time (s)')
            plt.savefig(output_path / 'compute_times.png')
            plt.close()

            # Generate HTML report
            self._generate_html_report(output_path)

        except Exception as e:
            self.logger.error(f"Error in visualization: {str(e)}")
            raise

    def _generate_html_report(self, output_path: Path) -> None:
        """Generate detailed HTML report with all profiling information"""
        try:
            analysis = self.analyze_model_architecture()
            results = self.get_profiling_results()

            html_content = f"""
            <html>
                <head>
                    <title>PyTorch Model Profiling Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                        .warning {{ color: red; }}
                        .metric {{ font-weight: bold; }}
                    </style>
                </head>
                <body>
                    <h1>PyTorch Model Profiling Report</h1>
                    
                    <div class="section">
                        <h2>Model Architecture</h2>
                        <p>Total Parameters: {analysis['total_parameters']:,}</p>
                        <p>Trainable Parameters: {analysis['trainable_parameters']:,}</p>
                    </div>

                    <div class="section">
                        <h2>Performance Metrics</h2>
                        <p>Peak GPU Memory: {max(p.gpu_memory for p in self.layer_profiles.values()):.2f} MB</p>
                        <p>Total Compute Time: {sum(p.compute_time for p in self.layer_profiles.values()):.2f} s</p>
                    </div>

                    <div class="section">
                        <h2>Optimization Suggestions</h2>
                        {''.join(f'<p class="warning">{s["suggestion"]}</p>' for s in analysis['optimization_suggestions'])}
                    </div>

                    <div class="section">
                        <h2>Layer Profiles</h2>
                        {''.join(f'<div class="metric">{name}: {profile.compute_time:.4f}s, {profile.gpu_memory:.2f}MB</div>'
                                 for name, profile in self.layer_profiles.items())}
                    </div>
                </body>
            </html>
            """

            with open(output_path / 'profiling_report.html', 'w') as f:
                f.write(html_content)

        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            raise

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()
