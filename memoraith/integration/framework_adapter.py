# memoraith/integration/framework_adapter.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import asyncio
from ..exceptions import MemoraithError

class FrameworkAdapter(ABC):
    """Base class for framework-specific adapters."""

    def __init__(self, model: Any):
        self.model = model
        self.data: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self._is_profiling = False
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Context manager entry."""
        await self.start_profiling()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop_profiling()
        if exc_type is not None:
            self.logger.error(f"Error during profiling: {exc_val}")
            return False
        return True

    @abstractmethod
    async def start_profiling(self) -> None:
        """Start profiling the model."""
        raise NotImplementedError

    @abstractmethod
    async def stop_profiling(self) -> None:
        """Stop profiling the model."""
        raise NotImplementedError

    @abstractmethod
    async def profile_inference(self, input_data: Any) -> Dict[str, Any]:
        """Profile model inference."""
        raise NotImplementedError

    @abstractmethod
    async def profile_training_step(self, input_data: Any, target: Any) -> Dict[str, Any]:
        """Profile a single training step."""
        raise NotImplementedError

    async def get_profiling_data(self) -> Dict[str, Any]:
        """Get collected profiling data."""
        async with self._lock:
            return self.data.copy()

    @abstractmethod
    async def get_model_summary(self) -> Dict[str, Any]:
        """Get model architecture summary."""
        raise NotImplementedError

    async def validate_model(self) -> bool:
        """Validate model compatibility."""
        try:
            # Basic validation
            if self.model is None:
                raise MemoraithError("Model is None")

            # Framework-specific validation should be implemented
            # in derived classes
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return False

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources."""
        raise NotImplementedError
