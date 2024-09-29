import logging
from typing import List, Optional
import asyncio

logger = logging.getLogger(__name__)

try:
    from pynvml import *
    PYNVML_AVAILABLE = True
except ImportError:
    logger.warning("pynvml not available. GPU memory tracking will be disabled.")
    PYNVML_AVAILABLE = False

from ..exceptions import GPUNotAvailableError

class GPUMemoryTracker:
    """Tracks GPU memory usage over time for NVIDIA GPUs."""

    def __init__(self, device_id: int = 0, interval: float = 0.1):
        self.device_id = device_id
        self.interval = interval
        self.memory_usage: List[float] = []
        self._stop_event: Optional[asyncio.Event] = None
        self._tracking_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start tracking GPU memory usage."""
        if not PYNVML_AVAILABLE:
            self.logger.warning("GPU memory tracking is not available.")
            return

        try:
            nvmlInit()
            self.device = nvmlDeviceGetHandleByIndex(self.device_id)
            self._stop_event = asyncio.Event()
            self._tracking_task = asyncio.create_task(self._track_memory())
            self.logger.info(f"Started GPU memory tracking for device {self.device_id}")
        except NVMLError as e:
            self.logger.error(f"Failed to initialize GPU memory tracking: {str(e)}")
            raise GPUNotAvailableError(f"GPU tracking failed: {str(e)}")

    async def stop(self) -> None:
        """Stop tracking GPU memory usage."""
        if not PYNVML_AVAILABLE:
            return

        if self._stop_event:
            self._stop_event.set()
        if self._tracking_task:
            await self._tracking_task
        try:
            nvmlShutdown()
            self.logger.info("Stopped GPU memory tracking")
        except NVMLError as e:
            self.logger.error(f"Error during NVML shutdown: {str(e)}")

    async def _track_memory(self) -> None:
        """Internal method to track memory usage."""
        while not self._stop_event.is_set():
            try:
                mem_info = nvmlDeviceGetMemoryInfo(self.device)
                self.memory_usage.append(mem_info.used / 1024**2)  # Convert to MB
            except NVMLError as e:
                self.logger.error(f"Error tracking GPU memory: {str(e)}")
            await asyncio.sleep(self.interval)

    async def get_peak_memory(self) -> float:
        """Get the peak GPU memory usage."""
        return max(self.memory_usage) if self.memory_usage else 0

    async def get_average_memory(self) -> float:
        """Get the average GPU memory usage."""
        return sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0

    async def get_current_memory(self) -> float:
        """Get the current GPU memory usage."""
        if not PYNVML_AVAILABLE:
            return 0

        try:
            mem_info = nvmlDeviceGetMemoryInfo(self.device)
            return mem_info.used / 1024**2  # Convert to MB
        except NVMLError as e:
            self.logger.error(f"Error getting current GPU memory: {str(e)}")
            return 0

    async def get_memory_history(self) -> List[float]:
        """Get the full history of memory usage."""
        return self.memory_usage

    async def get_device_info(self) -> dict:
        """Get information about the GPU device being tracked."""
        if not PYNVML_AVAILABLE:
            return {"error": "GPU information not available"}

        try:
            device_name = nvmlDeviceGetName(self.device).decode('utf-8')
            total_memory = nvmlDeviceGetMemoryInfo(self.device).total / 1024**2  # Convert to MB
            cuda_version = nvmlSystemGetCudaDriverVersion()
            return {
                "device_id": self.device_id,
                "device_name": device_name,
                "total_memory": total_memory,
                "cuda_version": f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
            }
        except NVMLError as e:
            self.logger.error(f"Error getting GPU device info: {str(e)}")
            return {"error": str(e)}

    async def reset(self) -> None:
        """Reset the memory usage history."""
        self.memory_usage = []

    def __del__(self):
        """Ensure NVML is shut down when the object is deleted."""
        if PYNVML_AVAILABLE:
            try:
                nvmlShutdown()
            except:
                pass  # Ignore errors during shutdown in destructor