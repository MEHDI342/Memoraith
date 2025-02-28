import torch
import torch.nn as nn
from torch.autograd import Variable
import psutil
import threading
import time
import logging
import json
import os
import asyncio
import gc
from typing import Dict, Any, List, Optional, Tuple, Set, Union, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
import numpy as np
from pathlib import Path
import queue
from collections import defaultdict, OrderedDict
import warnings

from ..exceptions import MemoraithError
from .framework_adapter import FrameworkAdapter

class ProfilingLevel(Enum):
    """Defines the profiling detail level for resource monitoring."""
    BASIC = "basic"          # Basic metrics only
    MEMORY = "memory"        # Memory-focused profiling
    COMPUTE = "compute"      # Computation-focused profiling
    FULL = "full"            # All metrics and features

@dataclass
class LayerProfile:
    """Detailed layer profiling information for comprehensive analysis."""
    name: str
    layer_type: str
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    parameters: int = 0
    cpu_memory: float = 0.0
    gpu_memory: float = 0.0
    compute_time: float = 0.0
    flops: int = 0
    backward_time: float = 0.0
    gradient_norm: float = 0.0
    activation_memory: float = 0.0
    buffer_memory: float = 0.0
    cuda_memory_allocated: float = 0.0
    cuda_memory_cached: float = 0.0
    cuda_utilization: float = 0.0
    peak_memory: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary format for serialization."""
        return {k: v for k, v in asdict(self).items()}

class PyTorchAdapter(FrameworkAdapter):
    """
    Advanced PyTorch model profiling adapter with enterprise-grade metrics collection.

    Provides comprehensive profiling capabilities for PyTorch models including:
    - High-precision memory tracking (CPU & GPU)
    - Layer-by-layer performance analysis
    - Computation time measurement with microsecond accuracy
    - FLOPs estimation for different layer types
    - Gradient flow analysis
    - Bottleneck detection and optimization recommendations
    - Real-time resource monitoring

    This adapter is designed for production environments with robust error handling,
    minimal overhead, and advanced analysis capabilities.
    """

    def __init__(
            self,
            model: nn.Module,
            level: ProfilingLevel = ProfilingLevel.FULL,
            log_dir: str = "profiling_logs",
            device: Optional[torch.device] = None
    ):
        """
        Initialize the PyTorch adapter with comprehensive profiling configuration.

        Args:
            model (nn.Module): The PyTorch model to profile
            level (ProfilingLevel): Profiling detail level
            log_dir (str): Directory for profiling logs
            device (torch.device): Device to run the model on
        """
        super().__init__(model)
        self.level = level
        self.log_dir = Path(log_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configure logging system
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Profiling data structures
        self.layer_profiles: Dict[str, LayerProfile] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.memory_traces: Dict[str, List[Any]] = defaultdict(list)
        self.gradient_history: Dict[str, List[float]] = defaultdict(list)
        self.activation_maps: Dict[str, Any] = {}
        self.bottlenecks: Set[str] = set()

        # Performance monitoring
        self._monitoring_queue = queue.Queue(maxsize=10000)  # Prevent unbounded growth
        self._stop_monitoring = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._is_profiling = False  # Track active profiling status

        # CUDA events for precise timing (allocated on demand)
        self.cuda_events: Dict[str, torch.cuda.Event] = {}

        # Thread-safety mechanisms
        self._hook_lock = threading.RLock()  # Reentrant lock for hook operations
        self._data_lock = threading.RLock()  # Reentrant lock for data access
        self._is_cuda_available = torch.cuda.is_available()  # Cache CUDA availability

        # Hook tracking
        self._original_hooks: Dict[str, Dict[str, Any]] = {}
        self._hook_call_stack: Set[str] = set()

        # Move model to specified device if needed
        if next(model.parameters(), None) is not None:
            current_device = next(model.parameters()).device
            if current_device != self.device:
                self.logger.info(f"Moving model from {current_device} to {self.device}")
                self.model.to(self.device)

        self.logger.info(f"PyTorch adapter initialized with profiling level: {level.value}")

    def _setup_logging(self) -> None:
        """
        Configure detailed logging for the adapter with rotating file handler.

        Sets up a sophisticated logging system with both console and file output,
        ensuring comprehensive debugging capabilities while preventing log file
        growth issues in production environments.
        """
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Create a formatter with detailed information
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            # Console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            # File handler with rotation
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    self.log_dir / "pytorch_profiling.log",
                    maxBytes=10 * 1024 * 1024,  # 10MB
                    backupCount=5
                )
            except ImportError:
                # Fallback to standard file handler if RotatingFileHandler is not available
                file_handler = logging.FileHandler(self.log_dir / "pytorch_profiling.log")

            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            # Configure logger if not already configured
            if not self.logger.handlers:
                self.logger.addHandler(console_handler)
                self.logger.addHandler(file_handler)

            self.logger.setLevel(logging.DEBUG)
            self.logger.debug("Logging system initialized for PyTorch adapter")

        except Exception as e:
            # Fallback logging to ensure we can at least capture critical errors
            print(f"Error setting up logging system: {str(e)}")
            self.logger.setLevel(logging.INFO)

    async def start_profiling(self) -> None:
        """
        Start profiling by attaching hooks and initializing monitoring.

        This method implements the required abstract method from FrameworkAdapter and
        initializes the complete profiling infrastructure for PyTorch models.

        Handles graceful initialization with proper error handling and resource management.
        """
        try:
            # Use lock to ensure thread safety for concurrent calls
            with self._data_lock:
                # Avoid duplicate profiling initialization
                if self._is_profiling:
                    self.logger.warning("Profiling already active - ignoring duplicate start request")
                    return

                self.logger.info(f"Starting PyTorch profiling on device: {self.device}")

                # Reset profiling data structures
                self.layer_profiles.clear()
                self.training_history.clear()
                self.memory_traces.clear()
                self.gradient_history.clear()
                self.activation_maps.clear()
                self.bottlenecks.clear()

                # Reset CUDA for accurate tracking
                if self._is_cuda_available:
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        self.logger.warning(f"Could not reset CUDA memory stats: {e}")

                # Attach hooks to model components
                self._attach_hooks()

                # Start performance monitoring
                self.start_monitoring()

                # Track metrics at profiling start
                with torch.no_grad():
                    # Record initial memory state
                    initial_cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                    initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if self._is_cuda_available else 0

                    self.data['initial_memory'] = {
                        'cpu': initial_cpu_memory,
                        'gpu': initial_gpu_memory,
                        'timestamp': time.time()
                    }

                self._is_profiling = True

            self.logger.info(f"PyTorch profiling started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start profiling: {str(e)}", exc_info=True)
            # Ensure clean state on failure
            await self.cleanup()
            raise MemoraithError(f"Failed to start PyTorch profiling: {str(e)}")

    async def stop_profiling(self) -> None:
        """
        Stop profiling by detaching hooks and finalizing metrics collection.

        Carefully shuts down all profiling components and ensures proper cleanup
        of resources while preserving collected metrics data.

        Handles graceful shutdown with comprehensive error handling.
        """
        if not self._is_profiling:
            self.logger.warning("Profiling not active - ignoring stop request")
            return

        try:
            self.logger.info("Stopping PyTorch profiling")

            # Use lock to ensure thread safety
            with self._data_lock:
                # Record final memory state
                final_cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
                final_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024) if self._is_cuda_available else 0

                self.data['final_memory'] = {
                    'cpu': final_cpu_memory,
                    'gpu': final_gpu_memory,
                    'timestamp': time.time()
                }

                # Calculate memory growth if we have initial measurements
                if 'initial_memory' in self.data:
                    initial = self.data['initial_memory']
                    self.data['memory_growth'] = {
                        'cpu': final_cpu_memory - initial['cpu'],
                        'gpu': final_gpu_memory - initial['gpu'],
                        'duration': time.time() - initial['timestamp']
                    }

                # Process collected data before stopping monitoring
                self._finalize_profiling_data()

                # Mark as not profiling before removing hooks to prevent new data collection
                self._is_profiling = False

            # Stop monitoring
            self.stop_monitoring()

            # Remove hooks from model (outside the lock to avoid deadlocks)
            self._remove_hooks()

            self.logger.info("PyTorch profiling stopped successfully")

        except Exception as e:
            self.logger.error(f"Error during profiling shutdown: {str(e)}", exc_info=True)
            # Still mark as stopped even on error
            self._is_profiling = False
            raise

    def _attach_hooks(self) -> None:
        """
        Attach comprehensive profiling hooks to model layers.

        Implements sophisticated hook architecture for forward and backward pass
        instrumentation with precise timing and memory tracking capabilities.

        Uses a thread-safe approach to hook management.
        """
        with self._hook_lock:
            self.logger.debug("Attaching profiling hooks to model layers")

            # Clear existing hooks if any
            if self._original_hooks:
                self._remove_hooks()

            # Initialize hook tracking
            self._hook_call_stack = set()
            self._original_hooks = {}

            # Track modules with hooks to avoid duplicates
            hook_count = 0

            for name, module in self.model.named_modules():
                # Skip if it's not a proper layer (e.g., Sequential container with no parameters)
                if sum(p.numel() for p in module.parameters()) == 0 and not hasattr(module, 'weight'):
                    continue

                # Forward pre-hook for input analysis
                pre_hook_handle = module.register_forward_pre_hook(self._forward_pre_hook(name))

                # Forward hook for output analysis
                forward_hook_handle = module.register_forward_hook(self._forward_hook(name))

                # Backward hook for gradient analysis if module has parameters
                backward_hook_handle = None
                if hasattr(module, 'weight') and module.weight is not None:
                    backward_hook_handle = module.register_full_backward_hook(self._backward_hook(name))

                # Store hook handles for later removal
                self._original_hooks[name] = {
                    'pre_hook': pre_hook_handle,
                    'forward_hook': forward_hook_handle,
                    'backward_hook': backward_hook_handle
                }

                # Create CUDA events for precise timing if available
                if self._is_cuda_available:
                    self.cuda_events[f"{name}_forward_start"] = torch.cuda.Event(enable_timing=True)
                    self.cuda_events[f"{name}_forward_end"] = torch.cuda.Event(enable_timing=True)
                    self.cuda_events[f"{name}_backward_start"] = torch.cuda.Event(enable_timing=True)
                    self.cuda_events[f"{name}_backward_end"] = torch.cuda.Event(enable_timing=True)

                hook_count += 1

            self.logger.info(f"Attached profiling hooks to {hook_count} modules")

    def _remove_hooks(self) -> None:
        """
        Remove all profiling hooks to restore model to original state.

        Ensures all hooks are properly removed to prevent memory leaks and
        unexpected behavior in subsequent model usage.

        Uses a thread-safe approach to hook management.
        """
        with self._hook_lock:
            self.logger.debug("Removing profiling hooks")

            # Remove all hook handles
            for name, hooks in self._original_hooks.items():
                for hook_type, handle in hooks.items():
                    if handle is not None:
                        handle.remove()

            # Clear hook references
            self._original_hooks.clear()
            self._hook_call_stack.clear()

            # Clear CUDA events to free memory
            self.cuda_events.clear()

            self.logger.info("All profiling hooks removed successfully")

    def _forward_pre_hook(self, name: str):
        """
        Pre-forward pass hook for input analysis.

        Args:
            name (str): Layer name for identification

        Returns:
            callable: Hook function that collects input metrics
        """
        def hook(module: nn.Module, input: Tuple[torch.Tensor]):
            # Skip if not in profiling mode or recursive call
            if not self._is_profiling or name in self._hook_call_stack:
                return

            # Mark layer as being processed to prevent recursive tracking
            self._hook_call_stack.add(name)

            if self.level != ProfilingLevel.BASIC:
                try:
                    # Start CUDA timing
                    if self._is_cuda_available and name in self.cuda_events:
                        event = self.cuda_events.get(f"{name}_forward_start")
                        if event:
                            event.record()

                    # Record input statistics if input tensor exists and has grad tracking
                    if input and len(input) > 0 and input[0] is not None:
                        # Handle potential non-tensor inputs
                        if isinstance(input[0], torch.Tensor):
                            # Calculate memory usage
                            input_size = input[0].element_size() * input[0].nelement()

                            with self._data_lock:
                                self.memory_traces[f"{name}_input"].append(input_size / (1024 * 1024))  # Convert to MB

                            # Track shapes for analysis
                            if name not in self.layer_profiles:
                                with self._data_lock:
                                    self.layer_profiles[name] = LayerProfile(
                                        name=name,
                                        layer_type=module.__class__.__name__,
                                        input_shape=tuple(input[0].shape),
                                        parameters=sum(p.numel() for p in module.parameters(recurse=False) if p is not None)
                                    )

                    # CUDA memory tracking
                    if self._is_cuda_available:
                        self._record_cuda_memory(name, "pre_forward")

                except Exception as e:
                    self.logger.error(f"Error in forward pre-hook for {name}: {str(e)}", exc_info=True)
                finally:
                    # Remove from call stack
                    self._hook_call_stack.remove(name)

        return hook

    def _forward_hook(self, name: str):
        """
        Post-forward pass hook for comprehensive layer analysis.

        Args:
            name (str): Layer name for identification

        Returns:
            callable: Hook function that collects output and performance metrics
        """
        def hook(module: nn.Module, input: Tuple[torch.Tensor], output: torch.Tensor):
            # Skip if not in profiling mode or recursive call
            if not self._is_profiling or name in self._hook_call_stack:
                return

            # Mark layer as being processed to prevent recursive tracking
            self._hook_call_stack.add(name)

            try:
                # End CUDA timing
                if self._is_cuda_available:
                    event_start = self.cuda_events.get(f"{name}_forward_start")
                    event_end = self.cuda_events.get(f"{name}_forward_end")
                    if event_start and event_end:
                        event_end.record()

                # Get or create profile entry
                with self._data_lock:
                    if name not in self.layer_profiles:
                        input_shape = None
                        if input and len(input) > 0 and isinstance(input[0], torch.Tensor):
                            input_shape = tuple(input[0].shape)

                        profile = LayerProfile(
                            name=name,
                            layer_type=module.__class__.__name__,
                            input_shape=input_shape,
                            output_shape=tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                            parameters=sum(p.numel() for p in module.parameters(recurse=False) if p is not None),
                            flops=self._calculate_flops(module, input[0] if input and len(input) > 0 else None, output)
                        )
                        self.layer_profiles[name] = profile
                    else:
                        profile = self.layer_profiles[name]
                        # Update output shape which wasn't available in pre_hook
                        profile.output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else None
                        profile.flops = self._calculate_flops(module, input[0] if input and len(input) > 0 else None, output)

                # Perform memory analysis if configured
                if self.level in (ProfilingLevel.MEMORY, ProfilingLevel.FULL):
                    self._analyze_memory(name, module, output, profile)

                # Perform computation analysis if configured and CUDA is available
                if self.level in (ProfilingLevel.COMPUTE, ProfilingLevel.FULL) and self._is_cuda_available:
                    if event_start and event_end:
                        self._analyze_computation(name, module, profile, event_start, event_end)

                # Store activation maps for visualization if they're not too large
                with self._data_lock:
                    if isinstance(output, torch.Tensor) and output.numel() < 1e6:  # Limit to ~8MB for float32
                        try:
                            # Store small subset for visualization
                            if output.dim() > 1:
                                # For multi-dimensional tensors, store a manageable slice
                                slice_size = min(8, output.shape[0])
                                sample = output[:slice_size].detach().cpu().numpy()
                            else:
                                # For 1D tensors, store first few elements
                                slice_size = min(100, output.shape[0])
                                sample = output[:slice_size].detach().cpu().numpy()

                            self.activation_maps[name] = {
                                'sample': sample.tolist(),
                                'shape': list(output.shape),
                                'full_size_mb': output.element_size() * output.numel() / (1024 * 1024)
                            }
                        except Exception as e:
                            self.logger.debug(f"Failed to store activation map for {name}: {e}")

                # Check for bottlenecks
                with self._data_lock:
                    self._check_bottlenecks(name, profile)

            except Exception as e:
                self.logger.error(f"Error in forward hook for {name}: {str(e)}", exc_info=True)
            finally:
                # Remove from call stack
                self._hook_call_stack.remove(name)

        return hook

    def _backward_hook(self, name: str):
        """
        Backward pass hook for gradient analysis.

        Args:
            name (str): Layer name for identification

        Returns:
            callable: Hook function that collects gradient metrics
        """
        def hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
            # Skip if not in profiling mode or recursive call
            if not self._is_profiling or name in self._hook_call_stack:
                return

            # Mark layer as being processed to prevent recursive tracking
            self._hook_call_stack.add(name)

            try:
                # Record gradient timing with CUDA events
                if self._is_cuda_available:
                    event_start = self.cuda_events.get(f"{name}_backward_start")
                    event_end = self.cuda_events.get(f"{name}_backward_end")
                    if event_start and event_end:
                        event_start.record()

                # Analyze gradients if computation profiling is enabled
                if self.level in (ProfilingLevel.COMPUTE, ProfilingLevel.FULL):
                    if grad_output and len(grad_output) > 0 and grad_output[0] is not None:
                        try:
                            grad_norm = grad_output[0].norm().item()

                            with self._data_lock:
                                self.gradient_history[name].append(grad_norm)

                                if name in self.layer_profiles:
                                    self.layer_profiles[name].gradient_norm = grad_norm
                        except Exception as e:
                            self.logger.debug(f"Error calculating gradient norm for {name}: {e}")

                # End event recording
                if self._is_cuda_available and event_start and event_end:
                    event_end.record()

                    # Record backward timing
                    with self._data_lock:
                        if name in self.layer_profiles:
                            try:
                                torch.cuda.synchronize()
                                backward_time_ms = event_start.elapsed_time(event_end)
                                self.layer_profiles[name].backward_time = backward_time_ms / 1000  # Convert to seconds
                            except Exception as e:
                                self.logger.debug(f"Error measuring backward time for {name}: {e}")

            except Exception as e:
                self.logger.error(f"Error in backward hook for {name}: {str(e)}", exc_info=True)
            finally:
                # Remove from call stack
                self._hook_call_stack.remove(name)

        return hook

    def _record_cuda_memory(self, name: str, stage: str) -> None:
        """
        Record CUDA memory usage at a specific profiling stage.

        Args:
            name (str): Layer name for identification
            stage (str): Current profiling stage (e.g., "pre_forward", "post_forward")
        """
        if not self._is_cuda_available:
            return

        try:
            # Get current memory statistics
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

            # Track device utilization if available (newer PyTorch versions)
            utilization = 0
            try:
                if hasattr(torch.cuda, 'utilization'):
                    utilization = torch.cuda.utilization()
            except:
                pass

            # Store memory metrics
            key = f"{name}_{stage}"
            memory_data = {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'utilization': utilization,
                'timestamp': time.time()
            }

            with self._data_lock:
                self.memory_traces[key].append(memory_data)

                # Update layer profile if it exists
                if name in self.layer_profiles:
                    profile = self.layer_profiles[name]
                    profile.cuda_memory_allocated = allocated
                    profile.cuda_memory_cached = reserved
                    profile.peak_memory = max(profile.peak_memory, max_allocated)
                    profile.cuda_utilization = utilization

            # Log detailed memory info at debug level only
            self.logger.debug(f"CUDA memory for {key}: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")

        except Exception as e:
            # Non-critical error, log but continue
            self.logger.debug(f"Error recording CUDA memory for {name}: {str(e)}")

    def _analyze_memory(self, name: str, module: nn.Module, output: torch.Tensor, profile: LayerProfile) -> None:
        """
        Detailed memory analysis for a layer.

        Args:
            name (str): Layer name
            module (nn.Module): Layer module
            output (torch.Tensor): Layer output
            profile (LayerProfile): Layer profile to update
        """
        try:
            # CPU memory via psutil
            profile.cpu_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

            # GPU memory if available
            if self._is_cuda_available:
                profile.gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                profile.cuda_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                profile.cuda_memory_cached = torch.cuda.memory_reserved() / (1024 * 1024)
                profile.peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Activation memory - the memory used by the output tensor
            if isinstance(output, torch.Tensor):
                profile.activation_memory = output.element_size() * output.nelement() / (1024 * 1024)  # MB

            # Buffer memory - the memory used by persistent buffers (e.g., running_mean in BatchNorm)
            buffer_memory = 0
            for buffer in module.buffers():
                if buffer is not None:
                    buffer_memory += buffer.element_size() * buffer.nelement()
            profile.buffer_memory = buffer_memory / (1024 * 1024)  # MB

        except Exception as e:
            # Non-critical error, log but continue
            self.logger.debug(f"Error in memory analysis for {name}: {str(e)}")

    def _analyze_computation(self, name: str, module: nn.Module, profile: LayerProfile,
                             event_start: torch.cuda.Event, event_end: torch.cuda.Event) -> None:
        """
        Detailed computation analysis for a layer.

        Args:
            name (str): Layer name
            module (nn.Module): Layer module
            profile (LayerProfile): Layer profile to update
            event_start (torch.cuda.Event): Start timing event
            event_end (torch.cuda.Event): End timing event
        """
        try:
            # CUDA timing with synchronization
            torch.cuda.synchronize()
            elapsed_time = event_start.elapsed_time(event_end)
            profile.compute_time = elapsed_time / 1000  # Convert from ms to seconds

            # CUDA utilization if available
            try:
                if hasattr(torch.cuda, 'utilization'):
                    profile.cuda_utilization = torch.cuda.utilization()
            except:
                pass

        except Exception as e:
            # Non-critical error, log but continue
            self.logger.debug(f"Error in computation analysis for {name}: {str(e)}")

    def _calculate_flops(self, module: nn.Module, input_tensor: Optional[torch.Tensor],
                         output_tensor: Optional[torch.Tensor]) -> int:
        """
        Calculate FLOPs (floating point operations) for different layer types.

        Args:
            module (nn.Module): Layer module
            input_tensor (torch.Tensor): Input tensor or None
            output_tensor (torch.Tensor): Output tensor or None

        Returns:
            int: Estimated number of FLOPs
        """
        try:
            if isinstance(module, nn.Conv2d) and input_tensor is not None:
                return self._conv2d_flops(module, input_tensor)
            elif isinstance(module, nn.Linear):
                return self._linear_flops(module)
            elif isinstance(module, nn.LSTM):
                return self._lstm_flops(module)
            elif isinstance(module, nn.LSTMCell):
                return self._lstm_cell_flops(module)
            elif isinstance(module, nn.GRU):
                return self._gru_flops(module)
            elif isinstance(module, nn.BatchNorm2d) and input_tensor is not None:
                return self._batchnorm2d_flops(module, input_tensor)
            # Add more layer types as needed
            return 0
        except Exception as e:
            # Non-critical error, log but continue
            self.logger.debug(f"Error calculating FLOPs: {str(e)}")
            return 0

    def _conv2d_flops(self, module: nn.Conv2d, input_tensor: torch.Tensor) -> int:
        """
        Calculate FLOPs for Conv2d layer with high precision.

        Args:
            module (nn.Conv2d): Convolution layer
            input_tensor (torch.Tensor): Input tensor

        Returns:
            int: Estimated number of FLOPs
        """
        # Guard against invalid inputs
        if not isinstance(input_tensor, torch.Tensor) or input_tensor.dim() != 4:
            return 0

        batch_size = input_tensor.shape[0]

        # Calculate output dimensions accounting for padding, dilation, and stride
        output_height = ((input_tensor.shape[2] + 2 * module.padding[0] - module.dilation[0] *
                          (module.kernel_size[0] - 1) - 1) // module.stride[0]) + 1
        output_width = ((input_tensor.shape[3] + 2 * module.padding[1] - module.dilation[1] *
                         (module.kernel_size[1] - 1) - 1) // module.stride[1]) + 1

        # Compute FLOPs per kernel element, considering groups
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups

        # Total FLOPs: 2 operations (multiply + add) per element
        flops = 2 * kernel_ops * output_height * output_width * batch_size * module.out_channels

        # Add bias FLOPs if present (one add per output element)
        if module.bias is not None:
            flops += output_height * output_width * batch_size * module.out_channels

        return flops

    def _linear_flops(self, module: nn.Linear) -> int:
        """
        Calculate FLOPs for Linear layer with high precision.

        Args:
            module (nn.Linear): Linear layer

        Returns:
            int: Estimated number of FLOPs
        """
        # Assume batch size of 1 for simplicity if not profiling a specific batch
        batch_size = 1

        # 2 FLOPs (multiply + add) per weight parameter
        flops = 2 * batch_size * module.in_features * module.out_features

        # Add FLOPs for bias if present (one add per output element)
        if module.bias is not None:
            flops += batch_size * module.out_features

        return flops

    def _lstm_flops(self, module: nn.LSTM) -> int:
        """
        Calculate FLOPs for LSTM layer with high precision.

        Args:
            module (nn.LSTM): LSTM layer

        Returns:
            int: Estimated number of FLOPs
        """
        # Default assumptions for FLOPs calculation
        batch_size = 1
        seq_length = 1

        # Calculate gates (input, forget, cell, output) FLOPs
        # Each gate: 2 * (input_size + hidden_size) * hidden_size FLOPs (matrix multiplication)
        gate_flops = 4 * 2 * (module.input_size + module.hidden_size) * module.hidden_size

        # Additional operations (activations, elementwise multiplications)
        # - 3 sigmoid activations (input, forget, output gates)
        # - 1 tanh activation (cell state)
        # - 1 tanh for output calculation
        # - 4 elementwise multiplications
        additional_ops = (3 * 4 + 2 * 4 + 4) * module.hidden_size  # Approximate FLOPs for these operations

        # Total FLOPs for all time steps, layers, and directions
        total_flops = (gate_flops + additional_ops) * batch_size * seq_length * module.num_layers

        # Account for bidirectional LSTM (doubles the computation)
        if module.bidirectional:
            total_flops *= 2

        return total_flops

    def _lstm_cell_flops(self, module: nn.LSTMCell) -> int:
        """
        Calculate FLOPs for LSTMCell layer with high precision.

        Args:
            module (nn.LSTMCell): LSTMCell layer

        Returns:
            int: Estimated number of FLOPs
        """
        batch_size = 1  # Default assumption

        # Gate calculations (input, forget, cell, output gates)
        # Each gate: 2 * (input_size + hidden_size) * hidden_size FLOPs
        gate_flops = 4 * 2 * (module.input_size + module.hidden_size) * module.hidden_size

        # Additional operations (similar to LSTM layer)
        additional_ops = (3 * 4 + 2 * 4 + 4) * module.hidden_size

        return (gate_flops + additional_ops) * batch_size

    def _gru_flops(self, module: nn.GRU) -> int:
        """
        Calculate FLOPs for GRU layer with high precision.

        Args:
            module (nn.GRU): GRU layer

        Returns:
            int: Estimated number of FLOPs
        """
        batch_size = 1
        seq_length = 1

        # GRU has 3 gates (reset, update, new)
        # Each gate: 2 * (input_size + hidden_size) * hidden_size FLOPs
        gate_flops = 3 * 2 * (module.input_size + module.hidden_size) * module.hidden_size

        # Additional operations:
        # - 2 sigmoid activations (reset, update gates)
        # - 1 tanh activation (new gate)
        # - 3 elementwise multiplications
        additional_ops = (2 * 4 + 1 * 4 + 3) * module.hidden_size

        total_flops = (gate_flops + additional_ops) * batch_size * seq_length * module.num_layers

        # Account for bidirectional GRU
        if module.bidirectional:
            total_flops *= 2

        return total_flops

    def _batchnorm2d_flops(self, module: nn.BatchNorm2d, input_tensor: torch.Tensor) -> int:
        """
        Calculate FLOPs for BatchNorm2d layer with high precision.

        Args:
            module (nn.BatchNorm2d): BatchNorm layer
            input_tensor (torch.Tensor): Input tensor

        Returns:
            int: Estimated number of FLOPs
        """
        # Guard against invalid inputs
        if not isinstance(input_tensor, torch.Tensor) or input_tensor.dim() != 4:
            return 0

        batch_size, channels, height, width = input_tensor.shape
        elements = batch_size * channels * height * width

        # BatchNorm operations per element:
        # 1. Subtract mean: (x - mean)
        # 2. Divide by standard deviation: (x - mean) / sqrt(var + eps)
        # 3. Multiply by gamma: gamma * ((x - mean) / sqrt(var + eps))
        # 4. Add beta: gamma * ((x - mean) / sqrt(var + eps)) + beta
        operations_per_element = 4

        return operations_per_element * elements

    def _check_bottlenecks(self, name: str, profile: LayerProfile) -> None:
        """
        Identify performance bottlenecks based on sophisticated heuristics.

        Implements advanced detection logic to find layers that might be
        causing performance issues based on memory usage, computation time,
        and gradient behavior.

        Args:
            name (str): Layer name
            profile (LayerProfile): Layer profile data
        """
        try:
            # Memory bottleneck detection with smart thresholds
            if profile.gpu_memory > 500:  # >500MB GPU memory usage
                severity = "critical" if profile.gpu_memory > 1000 else "high"
                self.bottlenecks.add(f"{name}_high_gpu_memory_{severity}")
                self.logger.debug(f"Memory bottleneck detected in {name}: {profile.gpu_memory:.2f}MB GPU usage")

            # Computation bottleneck detection - layers taking excessive time
            if profile.compute_time > 0.05:  # >50ms per layer invocation
                severity = "critical" if profile.compute_time > 0.2 else "high"
                self.bottlenecks.add(f"{name}_slow_compute_{severity}")
                self.logger.debug(f"Compute bottleneck detected in {name}: {profile.compute_time*1000:.2f}ms")

            # Gradient bottleneck detection - potential exploding/vanishing gradients
            if hasattr(profile, 'gradient_norm'):
                if profile.gradient_norm > 100:  # Potentially exploding gradients
                    self.bottlenecks.add(f"{name}_high_gradient_norm")
                    self.logger.debug(f"Gradient bottleneck (high): {name}, norm = {profile.gradient_norm:.2f}")
                elif profile.gradient_norm < 1e-3 and profile.gradient_norm > 0:  # Potentially vanishing gradients
                    self.bottlenecks.add(f"{name}_low_gradient_norm")
                    self.logger.debug(f"Gradient bottleneck (low): {name}, norm = {profile.gradient_norm:.6f}")

            # Computation efficiency bottlenecks - excessive FLOPs
            if profile.flops > 1e9:  # >1 GFLOPs for a single operation
                self.bottlenecks.add(f"{name}_high_flops")
                self.logger.debug(f"Computation efficiency bottleneck: {name}, {profile.flops/1e9:.2f} GFLOPs")

            # Activation memory bottlenecks - high memory for feature maps
            if profile.activation_memory > 100:  # >100MB for activations
                severity = "critical" if profile.activation_memory > 500 else "high"
                self.bottlenecks.add(f"{name}_high_activation_memory_{severity}")
                self.logger.debug(f"Activation memory bottleneck: {name}, {profile.activation_memory:.2f}MB")

            # Parameter efficiency bottlenecks
            if profile.parameters > 1e7:  # >10M parameters in a single layer
                self.bottlenecks.add(f"{name}_parameter_heavy")
                self.logger.debug(f"Parameter efficiency bottleneck: {name}, {profile.parameters} parameters")

        except Exception as e:
            self.logger.error(f"Error checking bottlenecks for {name}: {str(e)}")

    def start_monitoring(self) -> None:
        """
        Start continuous resource monitoring in background thread.

        Initializes background monitoring of system-wide resources including
        CPU, GPU, and memory metrics for correlation with model operations.
        """
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True,
            name="MemorithResourceMonitor"
        )
        self._monitoring_thread.start()
        self.logger.debug("Resource monitoring thread started")

    def stop_monitoring(self) -> None:
        """
        Stop resource monitoring and finalize metrics collection.

        Gracefully terminates the background monitoring thread and ensures
        all collected data is properly stored for analysis.
        """
        if not self._monitoring_thread:
            return

        self._stop_monitoring.set()
        if self._monitoring_thread.is_alive():
            try:
                self._monitoring_thread.join(timeout=2.0)  # Wait with timeout to avoid hanging
                if self._monitoring_thread.is_alive():
                    self.logger.warning("Resource monitoring thread didn't terminate cleanly")
            except Exception as e:
                self.logger.error(f"Error stopping monitoring thread: {str(e)}")

        self.logger.debug("Resource monitoring stopped")

    def _monitor_resources(self) -> None:
        """
        Continuous resource monitoring thread for background metric collection.

        Collects system-wide resource usage metrics at regular intervals
        to correlate with model operations and detect potential issues.
        """
        interval = 0.1  # 100ms sampling interval
        sample_count = 0
        throttle_log = 0  # Counter to throttle logging

        # Reduce monitoring overhead by limiting data collection to relevant metrics
        collect_gpu = self._is_cuda_available
        has_process_api = hasattr(psutil.Process(), "cpu_percent")

        # Enable adaptive interval based on system load
        adaptive_interval = interval
        min_interval = 0.1
        max_interval = 1.0

        while not self._stop_monitoring.is_set():
            try:
                start_time = time.time()

                # Get current CPU metrics with minimal system impact
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()

                # Get current GPU metrics if available (with error handling)
                gpu_info = {}
                if collect_gpu:
                    try:
                        for device_idx in range(torch.cuda.device_count()):
                            # Only collect essential metrics to minimize overhead
                            allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
                            reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)

                            gpu_info[f"gpu_{device_idx}"] = {
                                'memory_allocated_mb': allocated,
                                'memory_reserved_mb': reserved
                            }

                            # Collect utilization less frequently to reduce overhead
                            if sample_count % 5 == 0 and hasattr(torch.cuda, 'utilization'):
                                try:
                                    gpu_info[f"gpu_{device_idx}"]['utilization'] = torch.cuda.utilization(device_idx)
                                except:
                                    pass
                    except Exception as e:
                        if throttle_log % 100 == 0:  # Log only occasionally to prevent spam
                            self.logger.debug(f"Error collecting GPU metrics: {e}")
                        throttle_log += 1

                # Collect process metrics with minimal system impact
                process_metrics = {}
                if has_process_api:
                    try:
                        process = psutil.Process()
                        process_metrics = {
                            'memory_rss_mb': process.memory_info().rss / (1024 * 1024),
                            'cpu_percent': process.cpu_percent(interval=None)
                        }
                    except Exception as e:
                        if throttle_log % 100 == 0:
                            self.logger.debug(f"Error collecting process metrics: {e}")
                        throttle_log += 1

                # Record metrics with timestamp
                metrics = {
                    'timestamp': time.time(),
                    'cpu': {
                        'percent': cpu_percent,
                        'memory_used_percent': memory_info.percent,
                        'memory_available_mb': memory_info.available / (1024 * 1024)
                    },
                    'gpu': gpu_info,
                    'process': process_metrics
                }

                # Add to monitoring queue with overflow protection
                try:
                    # If queue is getting too large, remove oldest items to prevent memory issues
                    if self._monitoring_queue.qsize() > 5000:
                        try:
                            # Remove 20% of the oldest items
                            to_remove = min(1000, self._monitoring_queue.qsize() // 5)
                            for _ in range(to_remove):
                                self._monitoring_queue.get_nowait()

                            if throttle_log % 10 == 0:
                                self.logger.debug(f"Monitoring queue reached size limit, removed {to_remove} old samples")
                        except queue.Empty:
                            pass

                    self._monitoring_queue.put(metrics, block=False)
                    sample_count += 1
                except (queue.Full, BlockingIOError):
                    # Queue is full, skip this sample
                    pass

                # Log occasional status updates (but don't spam)
                if sample_count % 500 == 0:
                    self.logger.debug(f"Resource monitoring: {sample_count} samples collected")

                # Adaptive interval based on system load to minimize monitoring overhead
                execution_time = time.time() - start_time
                if execution_time > interval * 0.8:  # Monitoring is taking too long
                    adaptive_interval = min(adaptive_interval * 1.2, max_interval)
                elif sample_count > 100 and execution_time < interval * 0.2:  # Plenty of headroom
                    adaptive_interval = max(adaptive_interval * 0.8, min_interval)

                # Sleep for the adaptive interval, but ensure we don't sleep too long
                # if the stop signal was set during execution
                remaining_time = max(0.001, adaptive_interval - execution_time)
                time.sleep(remaining_time if not self._stop_monitoring.is_set() else 0.001)

            except Exception as e:
                if throttle_log % 100 == 0:
                    self.logger.error(f"Error in resource monitoring: {str(e)}")
                throttle_log += 1
                time.sleep(1.0)  # Wait longer on error to prevent log flooding

    def _get_monitoring_data(self) -> List[Dict[str, Any]]:
        """
        Retrieve collected monitoring data for analysis.

        Returns:
            List[Dict[str, Any]]: Collected resource monitoring data
        """
        data = []
        try:
            # Extract data from queue with a reasonable size limit
            max_items = 10000  # Prevent memory issues with extremely large datasets
            count = 0

            while not self._monitoring_queue.empty() and count < max_items:
                try:
                    data.append(self._monitoring_queue.get_nowait())
                    count += 1
                except queue.Empty:
                    break

            # Sort by timestamp to ensure chronological order
            if data:
                data.sort(key=lambda x: x.get('timestamp', 0))

        except Exception as e:
            self.logger.error(f"Error retrieving monitoring data: {str(e)}")

        return data

    def _finalize_profiling_data(self) -> None:
        """
        Process and finalize all collected profiling data.

        Aggregates metrics, calculates additional derived metrics,
        and prepares data for export and visualization.
        """
        self.logger.debug("Finalizing profiling data")

        try:
            # Process layer profiles
            for name, profile in self.layer_profiles.items():
                # Add profile to data dictionary for API consistency with other adapters
                self.data[f"layer_{name}"] = {
                    'name': name,
                    'type': profile.layer_type,
                    'parameters': profile.parameters,
                    'total_time': profile.compute_time,
                    'total_cpu_memory': profile.cpu_memory,
                    'total_gpu_memory': profile.gpu_memory,
                    'flops': profile.flops,
                    'activation_memory': profile.activation_memory,
                    'input_shape': profile.input_shape,
                    'output_shape': profile.output_shape,
                    'gradient_norm': profile.gradient_norm
                }

            # Add bottleneck information
            self.data['bottlenecks'] = [
                {
                    "layer": b.split('_')[0],
                    "type": '_'.join(b.split('_')[1:]),
                    "severity": "critical" if "critical" in b else "high" if "high" in b else "medium"
                }
                for b in self.bottlenecks
            ]

            # Process monitoring statistics
            monitoring_data = self._get_monitoring_data()
            if monitoring_data:
                # Calculate performance metrics
                cpu_usage = [sample['cpu']['percent'] for sample in monitoring_data]
                memory_usage = [sample['cpu']['memory_used_percent'] for sample in monitoring_data]
                process_memory = [sample['process'].get('memory_rss_mb', 0) for sample in monitoring_data if 'process' in sample]

                # GPU metrics if available
                gpu_memory = []
                for sample in monitoring_data:
                    for device, metrics in sample.get('gpu', {}).items():
                        if 'memory_allocated_mb' in metrics:
                            gpu_memory.append(metrics['memory_allocated_mb'])

                # Store summarized resource metrics
                self.data['resource_usage'] = {
                    'cpu_peak_percent': max(cpu_usage) if cpu_usage else 0,
                    'cpu_average_percent': sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                    'memory_peak_percent': max(memory_usage) if memory_usage else 0,
                    'memory_average_percent': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                    'process_peak_memory_mb': max(process_memory) if process_memory else 0,
                    'process_average_memory_mb': sum(process_memory) / len(process_memory) if process_memory else 0,
                    'gpu_peak_memory_mb': max(gpu_memory) if gpu_memory else 0,
                    'gpu_average_memory_mb': sum(gpu_memory) / len(gpu_memory) if gpu_memory else 0,
                    'sample_count': len(monitoring_data),
                    'monitoring_duration': (monitoring_data[-1]['timestamp'] - monitoring_data[0]['timestamp'])
                    if len(monitoring_data) > 1 else 0
                }

                # Store time series data for visualization (with sampling to reduce size)
                sample_rate = max(1, len(monitoring_data) // 1000)  # Limit to ~1000 data points
                self.data['resource_timeseries'] = {
                    'timestamps': [d['timestamp'] for d in monitoring_data[::sample_rate]],
                    'cpu_percent': [d['cpu']['percent'] for d in monitoring_data[::sample_rate]],
                    'memory_percent': [d['cpu']['memory_used_percent'] for d in monitoring_data[::sample_rate]],
                    'gpu_memory_mb': [
                        sum(m.get('memory_allocated_mb', 0) for m in d.get('gpu', {}).values())
                        for d in monitoring_data[::sample_rate]
                    ]
                }

            # Generate performance recommendations
            self.data['recommendations'] = self._generate_recommendations()

            # Calculate model-wide performance metrics
            total_params = sum(p.parameters for p in self.layer_profiles.values())
            total_flops = sum(p.flops for p in self.layer_profiles.values())
            total_time = sum(p.compute_time for p in self.layer_profiles.values())

            self.data['model_metrics'] = {
                'total_parameters': total_params,
                'total_flops': total_flops,
                'total_compute_time': total_time,
                'flops_per_second': total_flops / total_time if total_time > 0 else 0,
                'parameters_per_second': total_params / total_time if total_time > 0 else 0
            }

            self.logger.info(f"Profiling data finalized with {len(self.layer_profiles)} layer profiles")

        except Exception as e:
            self.logger.error(f"Error finalizing profiling data: {str(e)}", exc_info=True)

    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """
        Generate actionable optimization recommendations.

        Analyzes collected metrics and bottlenecks to provide targeted,
        actionable recommendations for model optimization.

        Returns:
            List[Dict[str, str]]: List of optimization recommendations
        """
        recommendations = []

        try:
            # Check for parameter-heavy layers
            param_heavy_layers = [(name, profile.parameters) for name, profile in self.layer_profiles.items()
                                  if profile.parameters > 5000000]  # >5M parameters

            param_heavy_layers.sort(key=lambda x: x[1], reverse=True)

            for name, params in param_heavy_layers[:3]:  # Top 3 parameter-heavy layers
                layer_type = self.layer_profiles[name].layer_type
                recommendations.append({
                    'layer': name,
                    'recommendation': (f"Consider reducing parameters in {layer_type} layer "
                                       f"(currently {params:,} parameters). "
                                       f"Options: reduce dimensions, use grouped convolutions, "
                                       f"apply model pruning, or knowledge distillation."),
                    'impact': 'high',
                    'category': 'memory'
                })

            # Check for compute-intensive layers
            compute_layers = [(name, profile.compute_time) for name, profile in self.layer_profiles.items()
                              if profile.compute_time > 0.01]  # >10ms

            compute_layers.sort(key=lambda x: x[1], reverse=True)

            for name, time_value in compute_layers[:3]:  # Top 3 compute-intensive layers
                layer_type = self.layer_profiles[name].layer_type
                flops = self.layer_profiles[name].flops

                # Tailor recommendations based on layer type
                if layer_type == 'Conv2d':
                    recommendation = (f"Optimize {layer_type} computation time (currently {time_value*1000:.2f}ms, "
                                      f"{flops/1e6:.1f}M FLOPs). Consider: reduce kernel size, "
                                      f"use grouped/depthwise convolutions, or reduce channels.")
                elif layer_type == 'Linear':
                    recommendation = (f"Optimize {layer_type} computation time (currently {time_value*1000:.2f}ms, "
                                      f"{flops/1e6:.1f}M FLOPs). Consider: reduce hidden dimensions, "
                                      f"apply matrix factorization, or use sparse connections.")
                elif layer_type in ('LSTM', 'GRU'):
                    recommendation = (f"Optimize {layer_type} computation time (currently {time_value*1000:.2f}ms, "
                                      f"{flops/1e6:.1f}M FLOPs). Consider: reduce hidden size, "
                                      f"use bidirectional only if necessary, or replace with 1D convolutions.")
                else:
                    recommendation = (f"Optimize {layer_type} computation time (currently {time_value*1000:.2f}ms, "
                                      f"{flops/1e6:.1f}M FLOPs). Consider more efficient operations or dimensions.")

                recommendations.append({
                    'layer': name,
                    'recommendation': recommendation,
                    'impact': 'high',
                    'category': 'compute'
                })

            # Check for high activation memory
            memory_layers = [(name, profile.activation_memory) for name, profile in self.layer_profiles.items()
                             if profile.activation_memory > 100]  # >100MB

            memory_layers.sort(key=lambda x: x[1], reverse=True)

            for name, memory_value in memory_layers[:3]:  # Top 3 memory-intensive layers
                layer_type = self.layer_profiles[name].layer_type
                input_shape = self.layer_profiles[name].input_shape
                output_shape = self.layer_profiles[name].output_shape

                shape_info = ""
                if input_shape and output_shape:
                    shape_info = f" Input shape: {input_shape}, Output shape: {output_shape}."

                recommendations.append({
                    'layer': name,
                    'recommendation': (f"High activation memory in {layer_type} ({memory_value:.1f}MB).{shape_info} "
                                       f"Consider: gradient checkpointing, reduced dimensions, or "
                                       f"feature map pruning."),
                    'impact': 'high',
                    'category': 'memory'
                })

            # Check for GPU memory usage
            if self._is_cuda_available:
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)  # GB
                if peak_memory > 4:
                    recommendations.append({
                        'global': 'memory',
                        'recommendation': (f"High GPU memory usage detected ({peak_memory:.2f}GB). "
                                           f"Consider: gradient checkpointing, reduced batch size, "
                                           f"or mixed precision training with torch.cuda.amp."),
                        'impact': 'critical',
                        'category': 'memory'
                    })

            # Check for potentially exploding/vanishing gradients
            high_grad_norms = [(name, max(values)) for name, values in self.gradient_history.items()
                               if values and max(values) > 100]

            if high_grad_norms:
                high_grad_norms.sort(key=lambda x: x[1], reverse=True)
                layer_names = ", ".join([name for name, _ in high_grad_norms[:3]])
                recommendations.append({
                    'global': 'training',
                    'recommendation': (f"High gradient norms detected in {layer_names}. "
                                       f"This may indicate exploding gradients. Consider: "
                                       f"gradient clipping, learning rate reduction, weight initialization "
                                       f"review, or adding batch normalization."),
                    'impact': 'high',
                    'category': 'training'
                })

            # Check for vanishing gradients
            low_grad_norms = [(name, min(values)) for name, values in self.gradient_history.items()
                              if values and 0 < min(values) < 1e-3]

            if low_grad_norms:
                low_grad_norms.sort(key=lambda x: x[1])
                layer_names = ", ".join([name for name, _ in low_grad_norms[:3]])
                recommendations.append({
                    'global': 'training',
                    'recommendation': (f"Very low gradient norms detected in {layer_names} "
                                       f"(potentially vanishing gradients). Consider: "
                                       f"residual connections, better initialization, "
                                       f"activation function review (e.g., use LeakyReLU instead of ReLU), "
                                       f"or layer normalization."),
                    'impact': 'high',
                    'category': 'training'
                })

            # Add general batch size recommendations based on available memory
            if hasattr(self.model, 'training') and self.model.training and self._is_cuda_available:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated()
                reserved_memory = torch.cuda.memory_reserved()
                free_memory = total_memory - reserved_memory
                free_memory_gb = free_memory / (1024 * 1024 * 1024)

                # Check if we can recommend increasing batch size
                if free_memory_gb > 4:
                    recommendations.append({
                        'global': 'performance',
                        'recommendation': (f"Consider increasing batch size to utilize available GPU memory "
                                           f"({free_memory_gb:.2f}GB free). Larger batches may improve "
                                           f"throughput and training stability."),
                        'impact': 'medium',
                        'category': 'performance'
                    })
                elif free_memory_gb < 0.5 and reserved_memory > allocated_memory * 1.5:
                    # We have fragmentation - reserved >> allocated
                    recommendations.append({
                        'global': 'memory',
                        'recommendation': (f"Detected memory fragmentation (reserved: {reserved_memory/(1024*1024*1024):.2f}GB, "
                                           f"allocated: {allocated_memory/(1024*1024*1024):.2f}GB). "
                                           f"Consider adding periodic torch.cuda.empty_cache() calls "
                                           f"or restructuring operations to reduce fragmentation."),
                        'impact': 'medium',
                        'category': 'memory'
                    })

            # Analyze model architecture for common inefficiencies
            sequential_count = sum(1 for _, module in self.model.named_modules() if isinstance(module, nn.Sequential))
            if sequential_count > 10:
                recommendations.append({
                    'global': 'architecture',
                    'recommendation': (f"Model uses many Sequential containers ({sequential_count}). "
                                       f"Consider flattening the architecture to reduce overhead."),
                    'impact': 'low',
                    'category': 'performance'
                })

            # Identify potential mixed precision opportunities
            if self._is_cuda_available and not hasattr(self.model, 'is_half'):
                has_complex_ops = any(m.__class__.__name__ in ['LSTM', 'GRU', 'RNN']
                                      for _, m in self.model.named_modules())
                if not has_complex_ops:
                    recommendations.append({
                        'global': 'performance',
                        'recommendation': ("Model is eligible for mixed precision training. "
                                           "Consider using torch.cuda.amp.autocast and GradScaler "
                                           "to potentially increase performance by up to 3x on modern GPUs."),
                        'impact': 'high',
                        'category': 'performance'
                    })

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
            return [{
                'global': 'error',
                'recommendation': "An error occurred while generating optimization recommendations.",
                'impact': 'none',
                'category': 'error'
            }]

        def __del__(self):
            """
            Ensure proper cleanup of resources when the adapter is garbage collected.

            Implements a failsafe cleanup mechanism for CUDA events and hooks.
            """
        try:
            # Cleanup hooks if they weren't already removed
            if hasattr(self, '_original_hooks') and self._original_hooks:
                try:
                    self._remove_hooks()
                except Exception as e:
                    if hasattr(self, 'logger'):
                        self.logger.warning(f"Error during hook cleanup in __del__: {e}")

            # Explicitly clear CUDA events dictionary
            if hasattr(self, 'cuda_events') and self.cuda_events:
                self.cuda_events.clear()

            # Force garbage collection to release CUDA resources
            if self._is_cuda_available:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # Ignore errors during cache clearing

            # Clear any remaining large data structures
            if hasattr(self, 'layer_profiles'):
                self.layer_profiles.clear()
            if hasattr(self, 'training_history'):
                self.training_history.clear()
            if hasattr(self, 'memory_traces'):
                self.memory_traces.clear()

        except Exception:
            # Never raise exceptions in __del__
            pass

    async def profile_inference(self, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        Profile a single inference pass with comprehensive metrics.

        Implements the abstract method from FrameworkAdapter with
        PyTorch-specific optimizations for inference profiling.

        Args:
            input_data (torch.Tensor): Input data for inference

        Returns:
            Dict[str, Any]: Detailed profiling metrics
        """
        local_profiling = False

        # Start profiling if not already active
        if not self._is_profiling:
            await self.start_profiling()
            local_profiling = True

        try:
            # Create context manager for consistent device placement
            device_context = (
                torch.cuda.device(self.device) if self._is_cuda_available and
                                                  self.device.type == 'cuda' else
                nullcontext()
            )

            # Move input to correct device if needed (with datatype preservation)
            if isinstance(input_data, torch.Tensor) and input_data.device != self.device:
                input_data = input_data.to(device=self.device,
                                           dtype=input_data.dtype,
                                           non_blocking=True)

            # Prepare model for inference
            original_training = self.model.training
            self.model.eval()

            # Reset performance metrics
            if self._is_cuda_available:
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.reset_accumulated_memory_stats(self.device)
                torch.cuda.synchronize(self.device)

            # Create CUDA events for precise timing if available
            start_event = end_event = None
            if self._is_cuda_available:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

            # Capture the start time
            start_cpu_time = time.perf_counter()
            if start_event:
                start_event.record()

            # Run the inference with gradients disabled
            with torch.no_grad(), device_context:
                output = self.model(input_data)

            # Record end time
            if end_event:
                end_event.record()
            end_cpu_time = time.perf_counter()

            # Sync CUDA to ensure timing accuracy
            if self._is_cuda_available:
                torch.cuda.synchronize(self.device)

            # Calculate timing metrics
            if self._is_cuda_available and start_event and end_event:
                inference_time = start_event.elapsed_time(end_event) / 1000  # ms to seconds
            else:
                inference_time = end_cpu_time - start_cpu_time

            # Collect memory data
            memory_data = {
                'cpu_memory': psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            }

            if self._is_cuda_available:
                memory_data.update({
                    'gpu_allocated': torch.cuda.memory_allocated(self.device) / (1024 * 1024),  # MB
                    'gpu_reserved': torch.cuda.memory_reserved(self.device) / (1024 * 1024),    # MB
                    'gpu_peak': torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)    # MB
                })

            # Calculate the throughput (samples per second)
            batch_size = input_data.shape[0] if isinstance(input_data, torch.Tensor) else 1
            throughput = batch_size / inference_time if inference_time > 0 else 0

            # Restore model training state
            self.model.train(original_training)

            # Create comprehensive profiling results
            profiling_data = {
                'inference_time': inference_time,
                'throughput': throughput,
                'batch_size': batch_size,
                'memory': memory_data,
                'input_shape': tuple(input_data.shape) if isinstance(input_data, torch.Tensor) else None,
                'output_shape': tuple(output.shape) if isinstance(output, torch.Tensor) else None,
                'device': str(self.device),
                'timestamp': time.time()
            }

            # Store in the adapter's data for API consistency
            key = f"inference_{len([k for k in self.data.keys() if k.startswith('inference_')])}"
            with self._data_lock:
                self.data[key] = profiling_data

            return profiling_data

        except Exception as e:
            self.logger.error(f"Error during inference profiling: {str(e)}", exc_info=True)
            raise MemoraithError(f"PyTorch inference profiling failed: {str(e)}")

        finally:
            # Stop profiling if we started it locally
            if local_profiling:
                await self.stop_profiling()

    async def profile_training_step(self,
                                    input_data: torch.Tensor,
                                    target: torch.Tensor,
                                    optimizer: torch.optim.Optimizer,
                                    criterion: nn.Module) -> Dict[str, float]:
        """
        Profile a single training step with detailed metrics.

        Implements the abstract method from FrameworkAdapter with
        PyTorch-specific optimizations for training step profiling.

        Args:
            input_data (torch.Tensor): Input data batch
            target (torch.Tensor): Target/label data
            optimizer (torch.optim.Optimizer): PyTorch optimizer
            criterion (nn.Module): Loss function module

        Returns:
            Dict[str, float]: Detailed profiling metrics for the training step
        """
        local_profiling = False

        # Start profiling if not already active
        if not self._is_profiling:
            await self.start_profiling()
            local_profiling = True

        try:
            # Create context manager for consistent device placement
            device_context = (
                torch.cuda.device(self.device) if self._is_cuda_available and
                                                  self.device.type == 'cuda' else
                nullcontext()
            )

            # Move data to correct device if needed (with non-blocking transfer)
            if isinstance(input_data, torch.Tensor) and input_data.device != self.device:
                input_data = input_data.to(device=self.device, non_blocking=True)

            if isinstance(target, torch.Tensor) and target.device != self.device:
                target = target.to(device=self.device, non_blocking=True)

            # Ensure model is in training mode
            self.model.train()

            # Reset performance metrics
            if self._is_cuda_available:
                torch.cuda.reset_peak_memory_stats(self.device)
                torch.cuda.synchronize(self.device)

            # Create CUDA events for precise phase timing
            events = {}
            if self._is_cuda_available:
                for phase in ['start', 'forward_start', 'forward_end',
                              'backward_start', 'backward_end',
                              'optimizer_start', 'optimizer_end', 'end']:
                    events[phase] = torch.cuda.Event(enable_timing=True)

            # Start overall timing
            cpu_timings = {'start': time.perf_counter()}
            if self._is_cuda_available:
                events['start'].record()

            # Clear gradients
            optimizer.zero_grad(set_to_none=True)  # More efficient than False

            # Forward pass with timing
            cpu_timings['forward_start'] = time.perf_counter()
            if self._is_cuda_available:
                events['forward_start'].record()

            with device_context:
                output = self.model(input_data)
                loss = criterion(output, target)

            cpu_timings['forward_end'] = time.perf_counter()
            if self._is_cuda_available:
                events['forward_end'].record()

            # Backward pass with timing
            cpu_timings['backward_start'] = time.perf_counter()
            if self._is_cuda_available:
                events['backward_start'].record()

            loss.backward()

            cpu_timings['backward_end'] = time.perf_counter()
            if self._is_cuda_available:
                events['backward_end'].record()

            # Optimizer step with timing
            cpu_timings['optimizer_start'] = time.perf_counter()
            if self._is_cuda_available:
                events['optimizer_start'].record()

            optimizer.step()

            cpu_timings['optimizer_end'] = time.perf_counter()
            if self._is_cuda_available:
                events['optimizer_end'].record()

            # End overall timing
            cpu_timings['end'] = time.perf_counter()
            if self._is_cuda_available:
                events['end'].record()
                torch.cuda.synchronize(self.device)

            # Calculate timing metrics (prefer CUDA events when available)
            if self._is_cuda_available:
                timings = {
                    'total': events['start'].elapsed_time(events['end']) / 1000,
                    'forward': events['forward_start'].elapsed_time(events['forward_end']) / 1000,
                    'backward': events['backward_start'].elapsed_time(events['backward_end']) / 1000,
                    'optimizer': events['optimizer_start'].elapsed_time(events['optimizer_end']) / 1000
                }
            else:
                timings = {
                    'total': cpu_timings['end'] - cpu_timings['start'],
                    'forward': cpu_timings['forward_end'] - cpu_timings['forward_start'],
                    'backward': cpu_timings['backward_end'] - cpu_timings['backward_start'],
                    'optimizer': cpu_timings['optimizer_end'] - cpu_timings['optimizer_start']
                }

            # Calculate gradient norms for all trainable parameters
            gradient_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    try:
                        gradient_norms[name] = param.grad.norm().item()
                    except:
                        # Skip parameters with unusual gradient types
                        pass

            # Estimate optimization efficiency
            op_efficiency = {
                'grad_to_param_ratio': {}  # Ratio of gradient to parameter magnitude
            }

            for name, param in self.model.named_parameters():
                if param.grad is not None and param.data.norm() > 0:
                    try:
                        op_efficiency['grad_to_param_ratio'][name] = param.grad.norm().item() / param.data.norm().item()
                    except:
                        # Skip parameters with unusual types
                        pass

            # Collect memory stats
            memory_stats = {
                'cpu_memory': psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            }

            if self._is_cuda_available:
                memory_stats.update({
                    'gpu_allocated': torch.cuda.memory_allocated(self.device) / (1024 * 1024),  # MB
                    'gpu_reserved': torch.cuda.memory_reserved(self.device) / (1024 * 1024),    # MB
                    'gpu_peak': torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)    # MB
                })

            # Calculate throughput metrics
            batch_size = input_data.shape[0] if isinstance(input_data, torch.Tensor) else 1
            samples_per_second = batch_size / timings['total']

            # Build comprehensive metrics dictionary
            metrics = {
                'loss': loss.item(),
                'timings': timings,
                'throughput': samples_per_second,
                'memory': memory_stats,
                'gradient_norms': gradient_norms,
                'optimization_efficiency': op_efficiency,
                'batch_size': batch_size,
                'device': str(self.device),
                'timestamp': time.time()
            }

            # Add to training history
            with self._data_lock:
                self.training_history.append(metrics)

                # Store in the adapter's data for API consistency
                key = f"training_step_{len([k for k in self.data.keys() if k.startswith('training_step_')])}"
                self.data[key] = metrics

            return metrics

        except Exception as e:
            self.logger.error(f"Error during training step profiling: {str(e)}", exc_info=True)
            raise MemoraithError(f"PyTorch training step profiling failed: {str(e)}")

        finally:
            # Stop profiling if we started it locally
            if local_profiling:
                await self.stop_profiling()

    async def get_model_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive model architecture summary.

        Implements the abstract method from FrameworkAdapter with
        PyTorch-specific optimizations for model analysis.

        Returns:
            Dict[str, Any]: Detailed model architecture summary
        """
        summary = {}

        try:
            # Total parameters with detailed breakdown
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            summary['total_params'] = total_params
            summary['trainable_params'] = trainable_params
            summary['frozen_params'] = frozen_params
            summary['param_distribution'] = {
                'trainable_percent': (trainable_params / total_params * 100) if total_params > 0 else 0
            }

            # Estimate model size in different precisions
            param_size_fp32 = total_params * 4 / (1024 * 1024)  # MB
            param_size_fp16 = total_params * 2 / (1024 * 1024)  # MB
            param_size_int8 = total_params * 1 / (1024 * 1024)  # MB

            summary['model_size'] = {
                'fp32_mb': param_size_fp32,
                'fp16_mb': param_size_fp16,
                'int8_mb': param_size_int8
            }

            # Analyze buffer memory
            buffer_params = sum(b.numel() for b in self.model.buffers())
            summary['buffer_params'] = buffer_params
            summary['buffer_size_mb'] = buffer_params * 4 / (1024 * 1024)  # Assuming float32

            # Layer breakdown with comprehensive metrics
            # Skip container modules with no parameters of their own
            layers_info = []
            for name, module in self.model.named_modules():
                # Skip parent modules that just contain other modules
                if sum(p.numel() for p in module.parameters(recurse=False)) == 0:
                    continue

                # Count parameters in this specific layer (not including children)
                params = sum(p.numel() for p in module.parameters(recurse=False))
                trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

                # Get input/output shapes from profiler if available
                input_shape = None
                output_shape = None
                flops = 0
                memory_usage = 0
                compute_time = 0

                if name in self.layer_profiles:
                    profile = self.layer_profiles[name]
                    input_shape = profile.input_shape
                    output_shape = profile.output_shape
                    flops = profile.flops
                    memory_usage = profile.gpu_memory if profile.gpu_memory > 0 else profile.cpu_memory
                    compute_time = profile.compute_time

                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'params': params,
                    'trainable': trainable,
                    'params_percent': params / total_params * 100 if total_params > 0 else 0,
                    'input_shape': input_shape,
                    'output_shape': output_shape,
                    'flops': flops,
                    'memory_mb': memory_usage,
                    'compute_time': compute_time
                }

                layers_info.append(layer_info)

            # Sort layers by parameter count for easier analysis
            layers_info.sort(key=lambda x: x['params'], reverse=True)
            summary['layers'] = layers_info

            # Analyze model structure
            module_types = {}
            for _, module in self.model.named_modules():
                module_type = module.__class__.__name__
                if module_type not in module_types:
                    module_types[module_type] = 0
                module_types[module_type] += 1

            summary['module_types'] = module_types

            # Device information
            if next(self.model.parameters(), None) is not None:
                summary['device'] = str(next(self.model.parameters()).device)
            else:
                summary['device'] = str(self.device)

            # Training mode
            summary['training_mode'] = self.model.training

            # Performance estimates based on profiling data
            if self.layer_profiles:
                total_flops = sum(p.flops for p in self.layer_profiles.values())
                total_compute_time = sum(p.compute_time for p in self.layer_profiles.values())

                if total_compute_time > 0:
                    summary['performance'] = {
                        'total_flops': total_flops,
                        'total_compute_time': total_compute_time,
                        'flops_per_second': total_flops / total_compute_time,
                        'theoretical_throughput': 1 / total_compute_time if total_compute_time > 0 else 0
                    }

            return summary

        except Exception as e:
            self.logger.error(f"Error generating model summary: {str(e)}", exc_info=True)
            return {'error': str(e)}

    async def validate_model(self) -> bool:
        """
        Validate the PyTorch model for profiling compatibility.

        Implements the abstract method from FrameworkAdapter with
        PyTorch-specific validation checks.

        Returns:
            bool: True if model is valid, False otherwise
        """
        try:
            # Check if it's a proper PyTorch model
            if not isinstance(self.model, nn.Module):
                self.logger.error("The provided model is not a PyTorch nn.Module")
                return False

            # Verify model has parameters
            param_count = sum(p.numel() for p in self.model.parameters())
            if param_count == 0:
                self.logger.warning("Model has no parameters - this may be an empty or incorrectly initialized model")

            # Verify model device
            device_params = next(self.model.parameters(), None)
            if device_params is not None:
                actual_device = device_params.device
                if actual_device != self.device:
                    self.logger.warning(f"Model is on {actual_device} but adapter configured for {self.device}")
                    # Attempt to move model to correct device
                    try:
                        self.model.to(self.device)
                        self.logger.info(f"Successfully moved model to {self.device}")
                    except Exception as e:
                        self.logger.error(f"Failed to move model to {self.device}: {e}")
                        return False

            # Check if the model has a valid forward method
            if not hasattr(self.model, 'forward'):
                self.logger.error("Model does not have a 'forward' method")
                return False

            # Verify forward method takes tensor input
            try:
                import inspect
                signature = inspect.signature(self.model.forward)
                if len(signature.parameters) < 1:
                    self.logger.warning("Model's forward method doesn't accept inputs")
            except:
                # Skip signature check if it fails
                pass

            # Verify CUDA is available if device is CUDA
            if self.device.type == 'cuda' and not self._is_cuda_available:
                self.logger.error("CUDA device specified but CUDA is not available")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """
        Clean up resources used during profiling.

        Implements the abstract method from FrameworkAdapter with
        PyTorch-specific resource cleanup operations.

        Ensures all resources are properly released to prevent memory leaks.
        """
        try:
            self.logger.debug("Cleaning up PyTorch adapter resources")

            # Stop monitoring if running
            self.stop_monitoring()

            # Remove all hooks
            self._remove_hooks()

            # Clear CUDA cache if available
            if self._is_cuda_available:
                try:
                    torch.cuda.empty_cache()
                    self.logger.debug("CUDA cache cleared")
                except Exception as e:
                    self.logger.warning(f"Failed to clear CUDA cache: {e}")

            # Clear intermediate data structures
            with self._data_lock:
                self.cuda_events.clear()
                self._hook_call_stack.clear()

                # Keep profiling results but clear intermediate data
                self.memory_traces.clear()
                self.activation_maps.clear()

            # Explicitly run garbage collection to free resources
            gc.collect()

            self.logger.info("PyTorch adapter cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during adapter cleanup: {str(e)}", exc_info=True)



class nullcontext:
    """
    A context manager that does NOTHING
    """
    def __enter__(self):
        return None

    def __exit__(self, *excinfo):
        pass