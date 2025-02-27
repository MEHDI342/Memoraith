import time
from typing import Dict, Optional, List
import threading
import logging

class TimeTracker:
    """Advanced time tracking for operations with nested timing support."""

    def __init__(self):
        self.start_times: Dict[str, List[float]] = {}
        self.end_times: Dict[str, List[float]] = {}
        self.durations: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    def start(self, key: str) -> None:
        """
        Start timing an operation. Supports nested timing for the same key.

        Args:
            key (str): Unique identifier for the operation
        """
        with self._lock:
            if key not in self.start_times:
                self.start_times[key] = []
                self.end_times[key] = []
                self.durations[key] = []
            self.start_times[key].append(time.perf_counter())
        self.logger.debug(f"Started timing for {key}")

    def stop(self, key: str) -> None:
        """
        Stop timing an operation. Matches the most recent start for the key.

        Args:
            key (str): Unique identifier for the operation
        """
        end_time = time.perf_counter()
        with self._lock:
            if key not in self.start_times or not self.start_times[key]:
                raise ValueError(f"No matching start time found for key: {key}")
            start_time = self.start_times[key].pop()
            self.end_times[key].append(end_time)
            duration = end_time - start_time
            self.durations[key].append(duration)
        self.logger.debug(f"Stopped timing for {key}. Duration: {duration:.6f} seconds")

    def get_duration(self, key: str) -> Optional[float]:
        """
        Get the total duration of all timings for a key.

        Args:
            key (str): Unique identifier for the operation

        Returns:
            Optional[float]: Total duration in seconds, or None if not available
        """
        with self._lock:
            if key in self.durations:
                return sum(self.durations[key])
        return None

    def get_average_duration(self, key: str) -> Optional[float]:
        """
        Get the average duration of all timings for a key.

        Args:
            key (str): Unique identifier for the operation

        Returns:
            Optional[float]: Average duration in seconds, or None if not available
        """
        with self._lock:
            if key in self.durations and self.durations[key]:
                return sum(self.durations[key]) / len(self.durations[key])
        return None

    def reset(self) -> None:
        """Reset all timings."""
        with self._lock:
            self.start_times.clear()
            self.end_times.clear()
            self.durations.clear()
        self.logger.info("All timings have been reset")

    def get_all_durations(self) -> Dict[str, List[float]]:
        """
        Get durations for all tracked operations.

        Returns:
            Dict[str, List[float]]: Dictionary of operation keys and their durations
        """
        with self._lock:
            return {key: durations.copy() for key, durations in self.durations.items()}

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of all tracked operations.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary with total, average, min, and max durations for each key
        """
        summary = {}
        with self._lock:
            for key, durations in self.durations.items():
                if durations:
                    summary[key] = {
                        'total': sum(durations),
                        'average': sum(durations) / len(durations),
                        'min': min(durations),
                        'max': max(durations),
                        'count': len(durations)
                    }
        return summary
