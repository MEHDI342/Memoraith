# memoraith/data_collection/base_collector.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging
from ..exceptions import DataCollectionError

class BaseDataCollector(ABC):
    """Base class for all data collectors."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.logger = logging.getLogger(__name__)
        self.data: Dict[str, Any] = {}
        self._is_collecting = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start data collection."""
        try:
            async with self._lock:
                if self._is_collecting:
                    raise DataCollectionError("Collection already running", details="Data collection process is already active")
                self._is_collecting = True
                self._task = asyncio.create_task(self._collection_loop())
                self.logger.info(f"{self.__class__.__name__} started collecting data")
        except Exception as e:
            self.logger.error(f"Failed to start collection: {str(e)}")
            raise DataCollectionError("Start failed", details=str(e))

    async def stop(self) -> None:
        """Stop data collection."""
        try:
            async with self._lock:
                if not self._is_collecting:
                    return
                self._is_collecting = False
                if self._task:
                    self._task.cancel()
                    try:
                        await self._task
                    except asyncio.CancelledError:
                        pass
                    self._task = None
                self.logger.info(f"{self.__class__.__name__} stopped collecting data")
        except Exception as e:
            self.logger.error(f"Failed to stop collection: {str(e)}")
            raise DataCollectionError("Stop failed", details=str(e))

    @abstractmethod
    async def _collect_data(self) -> Dict[str, Any]:
        """Collect a single data point."""
        raise NotImplementedError

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._is_collecting:
            try:
                data_point = await self._collect_data()
                async with self._lock:
                    self._process_data_point(data_point)
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in collection loop: {str(e)}")

    def _process_data_point(self, data_point: Dict[str, Any]) -> None:
        """Process and store a collected data point."""
        timestamp = data_point.get('timestamp', 0)
        if timestamp not in self.data:
            self.data[timestamp] = data_point

    async def get_data(self) -> Dict[str, Any]:
        """Get collected data."""
        async with self._lock:
            return self.data.copy()

    async def clear_data(self) -> None:
        """Clear collected data."""
        async with self._lock:
            self.data.clear()

    @abstractmethod
    async def validate_data(self) -> bool:
        """Validate collected data."""
        raise NotImplementedError