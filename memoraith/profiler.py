from typing import Dict, Any
import logging
import asyncio
from memoraith.data_collection.cpu_memory import CPUMemoryTracker
from memoraith.data_collection.gpu_memory import GPUMemoryTracker
from memoraith.data_collection.time_tracking import TimeTracker
from memoraith.data_collection.network_profiler import NetworkProfiler

class ModelProfiler:
    def __init__(self):
        self.cpu_tracker = CPUMemoryTracker()
        self.gpu_tracker = GPUMemoryTracker()
        self.time_tracker = TimeTracker()
        self.network_profiler = NetworkProfiler()
        self.logger = logging.getLogger(__name__)

    async def start_profiling(self):
        self.logger.info("Starting model profiling")
        await self.cpu_tracker.start()
        if self.gpu_tracker:
            await self.gpu_tracker.start()
        self.time_tracker.start('training')
        self.network_profiler.start()

    async def stop_profiling(self):
        self.logger.info("Stopping model profiling")
        cpu_memory = await self.cpu_tracker.get_peak_memory()
        gpu_memory = await self.gpu_tracker.get_peak_memory() if self.gpu_tracker else None
        duration = self.time_tracker.get_duration('training')
        network_usage = self.network_profiler.stop()

        profiling_results = {
            'cpu_memory': cpu_memory,
            'gpu_memory': gpu_memory,
            'training_time': duration,
            'network_usage': network_usage
        }
        self.logger.info(f"Profiling results: {profiling_results}")
        return profiling_results

    async def profile_step(self, step_name: str):
        self.time_tracker.start(step_name)
        cpu_memory_before = await self.cpu_tracker.get_current_memory()
        gpu_memory_before = await self.gpu_tracker.get_current_memory() if self.gpu_tracker else None
        network_usage_before = self.network_profiler.get_current_usage()

        yield  # Yield control to allow the step to execute

        cpu_memory_after = await self.cpu_tracker.get_current_memory()
        gpu_memory_after = await self.gpu_tracker.get_current_memory() if self.gpu_tracker else None
        network_usage_after = self.network_profiler.get_current_usage()
        duration = self.time_tracker.stop(step_name)

        step_profile = {
            'name': step_name,
            'duration': duration,
            'cpu_memory_used': cpu_memory_after - cpu_memory_before,
            'gpu_memory_used': gpu_memory_after - gpu_memory_before if gpu_memory_after and gpu_memory_before else None,
            'network_sent': network_usage_after['bytes_sent'] - network_usage_before['bytes_sent'],
            'network_recv': network_usage_after['bytes_recv'] - network_usage_before['bytes_recv'],
        }

        self.logger.info(f"Step profile for {step_name}: {step_profile}")
        yield step_profile

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_time': self.time_tracker.get_total_duration(),
            'peak_cpu_memory': self.cpu_tracker.get_peak_memory(),
            'peak_gpu_memory': self.gpu_tracker.get_peak_memory() if self.gpu_tracker else None,
            'average_network_usage': self.network_profiler.get_average_usage(),
        }

    def reset(self):
        self.cpu_tracker.reset()
        if self.gpu_tracker:
            self.gpu_tracker.reset()
        self.time_tracker.reset()
        self.network_profiler.reset()
        self.logger.info("All profilers reset")