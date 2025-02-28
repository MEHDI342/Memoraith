# memoraith/data_collection/base_collector.py
import time
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging
import traceback
from ..exceptions import DataCollectionError

class BaseDataCollector(ABC):
    """Base class for all data collectors with enterprise-grade error handling."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.logger = logging.getLogger(__name__)
        self.data: Dict[str, Any] = {}
        self._is_collecting = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Start data collection with comprehensive error handling.

        Raises:
            DataCollectionError: If collection is already running or initialization fails
        """
        try:
            async with self._lock:
                if self._is_collecting:
                    raise DataCollectionError("Collection already running", details="Data collection process is already active")

                # Set state before potential failure points to ensure cleanup
                self._is_collecting = True

                try:
                    self._task = asyncio.create_task(self._collection_loop())
                    self.logger.info(f"{self.__class__.__name__} started collecting data")
                except Exception as e:
                    # Reset state if task creation fails
                    self._is_collecting = False
                    self.logger.error(f"Failed to create collection task: {str(e)}")
                    # Preserve original exception chain
                    raise DataCollectionError("Start failed", details=str(e)) from e
        except Exception as e:
            if not isinstance(e, DataCollectionError):
                stack_trace = traceback.format_exc()
                self.logger.error(f"Failed to start collection: {str(e)}\n{stack_trace}")
                raise DataCollectionError("Start failed", details=str(e)) from e
            raise

    async def stop(self) -> None:
        """
        Stop data collection with graceful cleanup.

        Raises:
            DataCollectionError: If stopping collection fails
        """
        try:
            async with self._lock:
                if not self._is_collecting:
                    return

                self._is_collecting = False

                if self._task:
                    try:
                        self._task.cancel()
                        try:
                            await asyncio.wait_for(self._task, timeout=2.0)
                        except asyncio.TimeoutError:
                            self.logger.warning(f"{self.__class__.__name__} task cancellation timed out")
                        except asyncio.CancelledError:
                            pass  # Expected during cancellation
                    except Exception as e:
                        self.logger.error(f"Error during task cancellation: {str(e)}")
                    finally:
                        self._task = None

                self.logger.info(f"{self.__class__.__name__} stopped collecting data")
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.logger.error(f"Failed to stop collection: {str(e)}\n{stack_trace}")
            if not isinstance(e, DataCollectionError):
                raise DataCollectionError("Stop failed", details=str(e)) from e
            raise

    @abstractmethod
    async def _collect_data(self) -> Dict[str, Any]:
        """
        Collect a single data point.

        Returns:
            Dict[str, Any]: Collected data point with timestamp

        Raises:
            Exception: Implementation-specific exceptions
        """
        raise NotImplementedError

    async def _collection_loop(self) -> None:
        """
        Main collection loop with adaptive error handling.

        Implements exponential backoff for persistent errors.
        """
        error_count = 0
        backoff_time = 0.1
        max_backoff = 5.0  # Maximum backoff in seconds

        while self._is_collecting:
            try:
                data_point = await self._collect_data()

                async with self._lock:
                    self._process_data_point(data_point)

                # Reset error tracking after successful collection
                if error_count > 0:
                    error_count = 0
                    backoff_time = 0.1

                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                self.logger.debug(f"{self.__class__.__name__} collection loop cancelled")
                break
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error in collection loop ({error_count}): {str(e)}")

                # Implement exponential backoff for persistent errors
                if error_count > 10:
                    self.logger.critical(f"Too many errors in collection loop, aborting collection")
                    self._is_collecting = False
                    break

                # Calculate backoff time with exponential increase and jitter
                backoff_time = min(max_backoff, backoff_time * 1.5)
                jitter = backoff_time * 0.1 * (0.5 - random.random())  # Â±10% jitter
                await asyncio.sleep(backoff_time + jitter)

    def _process_data_point(self, data_point: Dict[str, Any]) -> None:
        """
        Process and store a collected data point with deduplication.

        Args:
            data_point: The data point to process and store
        """
        if not data_point:
            return

        timestamp = data_point.get('timestamp', 0)
        if timestamp == 0:
            self.logger.warning("Data point missing timestamp, using current time")
            timestamp = time.time()
            data_point['timestamp'] = timestamp

        # Use timestamp as key for deduplication
        if timestamp not in self.data:
            self.data[timestamp] = data_point
        else:
            # Merge with existing data point for same timestamp
            self.data[timestamp].update(data_point)

    async def get_data(self) -> Dict[str, Any]:
        """
        Get collected data with thread-safety.

        Returns:
            Dict[str, Any]: Copy of collected data
        """
        async with self._lock:
            return self.data.copy()

    async def clear_data(self) -> None:
        """Clear collected data with thread-safety."""
        async with self._lock:
            self.data.clear()
            self.logger.debug(f"{self.__class__.__name__} data cleared")

    @abstractmethod
    async def validate_data(self) -> bool:
        """
        Validate collected data for consistency and completeness.

        Returns:
            bool: True if data is valid, False otherwise
        """
        raise NotImplementedError