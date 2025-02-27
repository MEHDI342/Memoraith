import threading
from contextlib import contextmanager
import time
import logging
from typing import Optional

class ResourceLock:
    """
    Advanced resource locking mechanism with timeout and deadlock detection.
    """

    def __init__(self, name: str, timeout: float = 5.0):
        self._lock = threading.Lock()
        self._owner: Optional[int] = None
        self.name = name
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def __call__(self, timeout: Optional[float] = None):
        acquired = self.acquire(timeout)
        if not acquired:
            raise TimeoutError(f"Failed to acquire lock '{self.name}' within the specified timeout.")
        try:
            yield
        finally:
            self.release()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Attempt to acquire the lock with a specified timeout.

        Args:
            timeout (Optional[float]): The maximum time to wait for the lock. If None, use the default timeout.

        Returns:
            bool: True if the lock was acquired, False otherwise.
        """
        start_time = time.time()
        timeout = timeout or self.timeout

        while True:
            if self._lock.acquire(blocking=False):
                self._owner = threading.get_ident()
                self.logger.debug(f"Lock '{self.name}' acquired by thread {self._owner}")
                return True

            if time.time() - start_time > timeout:
                self.logger.warning(f"Timeout while attempting to acquire lock '{self.name}'")
                return False

            time.sleep(0.1)

    def release(self) -> None:
        """Release the lock."""
        if self._owner != threading.get_ident():
            raise RuntimeError(f"Attempt to release lock '{self.name}' by non-owner thread")

        self._owner = None
        self._lock.release()
        self.logger.debug(f"Lock '{self.name}' released by thread {threading.get_ident()}")

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock '{self.name}' within the specified timeout.")

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    @property
    def locked(self) -> bool:
        """Check if the lock is currently held."""
        return self._lock.locked()

    @property
    def owner(self) -> Optional[int]:
        """Get the ID of the thread that currently holds the lock, if any."""
        return self._owner
