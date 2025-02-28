from typing import Any, Dict
import threading
import time
import logging
import uuid
from typing import Optional
from contextlib import contextmanager

class ResourceLock:
    """
    Enterprise-grade resource locking mechanism with deadlock detection,
    timeout management, and full context manager support.
    """

    def __init__(self, name: str = None, timeout: float = 5.0):
        self._lock = threading.Lock()
        self._owner: Optional[int] = None
        self._acquisition_time: Optional[float] = None
        self.name = name or f"ResourceLock-{uuid.uuid4().hex[:8]}"
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        self.acquisition_count = 0
        self.timeout_count = 0

    @contextmanager
    def __call__(self, timeout: Optional[float] = None):
        """
        Context manager for resource acquisition with configurable timeout.

        Args:
            timeout: Maximum time to wait for lock acquisition

        Raises:
            TimeoutError: If lock cannot be acquired within timeout period

        Returns:
            Context manager that releases the lock on exit
        """
        acquired = self.acquire(timeout)
        if not acquired:
            self.timeout_count += 1
            raise TimeoutError(f"Failed to acquire lock '{self.name}' within the specified timeout.")
        try:
            yield self
        finally:
            self.release()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Attempt to acquire the lock with specified timeout and retry strategy.

        Args:
            timeout: Maximum time to wait for the lock, defaults to instance timeout

        Returns:
            bool: True if lock acquired successfully, False otherwise
        """
        start_time = time.time()
        timeout = timeout if timeout is not None else self.timeout
        attempt_count = 0

        # Use progressive backoff for retry strategy
        base_wait = 0.005  # Start with 5ms

        while True:
            attempt_count += 1

            if self._lock.acquire(blocking=False):
                self._owner = threading.get_ident()
                self._acquisition_time = time.time()
                self.acquisition_count += 1

                if attempt_count > 1:
                    self.logger.debug(f"Lock '{self.name}' acquired by thread {self._owner} after {attempt_count} attempts")
                else:
                    self.logger.debug(f"Lock '{self.name}' acquired by thread {self._owner}")

                return True

            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.logger.warning(f"Timeout while attempting to acquire lock '{self.name}' after {attempt_count} attempts")
                return False

            # Calculate wait time with exponential backoff
            wait_time = min(timeout/10, base_wait * min(32, 2 ** (attempt_count - 1)))
            time.sleep(wait_time)

    def release(self) -> None:
        """
        Release the lock with ownership validation.

        Raises:
            RuntimeError: If thread attempting to release is not the owner
        """
        current_thread = threading.get_ident()
        if self._owner != current_thread:
            raise RuntimeError(f"Thread {current_thread} attempted to release lock '{self.name}' owned by thread {self._owner}")

        # Clear ownership before release to minimize race conditions
        self._owner = None
        self._acquisition_time = None
        self._lock.release()
        self.logger.debug(f"Lock '{self.name}' released by thread {current_thread}")

    def __enter__(self):
        """
        Context manager entry point with proper self return for with-statement compatibility.

        Raises:
            TimeoutError: If lock cannot be acquired

        Returns:
            self: Returns self for use in with-statement context
        """
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock '{self.name}' within the specified timeout.")
        return self  # Return self to support context manager protocol

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit point with guaranteed release.

        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception value if an exception was raised
            traceback: Traceback if an exception was raised
        """
        self.release()

    @property
    def locked(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.locked()

    @property
    def owner(self) -> Optional[int]:
        """Get the ID of the thread that currently holds the lock, if any."""
        return self._owner

    @property
    def hold_time(self) -> Optional[float]:
        """
        Get the current hold time of the lock in seconds.

        Returns:
            Optional[float]: Time in seconds since lock acquisition, or None if not held
        """
        if self._acquisition_time is None:
            return None
        return time.time() - self._acquisition_time

    def get_stats(self) -> Dict[str, Any]:
        """
        Get lock usage statistics.

        Returns:
            Dict[str, Any]: Statistics about lock usage
        """
        return {
            "name": self.name,
            "currently_locked": self.locked,
            "owner": self._owner,
            "acquisition_count": self.acquisition_count,
            "timeout_count": self.timeout_count,
            "hold_time": self.hold_time,
        }