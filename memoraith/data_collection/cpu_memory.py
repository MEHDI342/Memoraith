import psutil
import threading
import time
from typing import List, Dict, Any
import logging

class CPUMemoryTracker:
    """
    Advanced CPU memory usage tracker with detailed memory breakdown.
    """

    def __init__(self, interval: float = 0.1, detailed: bool = True):
        self.interval = interval
        self.detailed = detailed
        self.memory_usage: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread = None
        self.logger = logging.getLogger(__name__)

    def start(self) -> None:
        """Start tracking CPU memory usage."""
        self._thread = threading.Thread(target=self._track_memory, daemon=True)
        self._thread.start()
        self.logger.info("CPU memory tracking started")

    def stop(self) -> None:
        """Stop tracking CPU memory usage."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        self.logger.info("CPU memory tracking stopped")

    def _track_memory(self) -> None:
        process = psutil.Process()
        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                mem_data = {
                    'timestamp': time.time(),
                    'rss': mem_info.rss / (1024 * 1024),  # RSS in MB
                    'vms': mem_info.vms / (1024 * 1024),  # VMS in MB
                }

                if self.detailed:
                    mem_maps = process.memory_maps(grouped=True)
                    mem_data.update({
                        'shared': sum(m.shared for m in mem_maps) / (1024 * 1024),
                        'private': sum(m.private for m in mem_maps) / (1024 * 1024),
                        'swap': sum(m.swap for m in mem_maps) / (1024 * 1024),
                    })

                with self._lock:
                    self.memory_usage.append(mem_data)
            except Exception as e:
                self.logger.error(f"Error tracking CPU memory: {str(e)}")

            time.sleep(self.interval)

    def get_peak_memory(self) -> Dict[str, float]:
        """Get the peak memory usage."""
        with self._lock:
            if not self.memory_usage:
                return {}
            return max(self.memory_usage, key=lambda x: x['rss'])

    def get_average_memory(self) -> Dict[str, float]:
        """Get the average memory usage."""
        with self._lock:
            if not self.memory_usage:
                return {}
            avg_mem = {key: sum(m[key] for m in self.memory_usage) / len(self.memory_usage)
                       for key in self.memory_usage[0] if key != 'timestamp'}
            return avg_mem

    def get_memory_timeline(self) -> List[Dict[str, Any]]:
        """Get the full timeline of memory usage."""
        with self._lock:
            return self.memory_usage.copy()

    def reset(self) -> None:
        """Reset the memory usage data."""
        with self._lock:
            self.memory_usage.clear()
        self.logger.info("CPU memory tracking data reset")