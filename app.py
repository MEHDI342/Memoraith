import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from .config import config
from .exceptions import MemoraithError
from .profiler import profile_model, set_output_path

class Memoraith:
    """Main application class for Memoraith profiler."""

    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        if config_file:
            config.load_from_file(config_file)

    async def setup(self) -> None:
        """Initialize the profiler setup."""
        try:
            # Ensure output directory exists
            config.output_path.mkdir(parents=True, exist_ok=True)

            # Validate configuration
            if not config.validate():
                raise MemoraithError("Invalid configuration")

            self.logger.info("Memoraith setup completed successfully")
        except Exception as e:
            self.logger.error(f"Setup failed: {str(e)}")
            raise

    @profile_model()
    async def profile(self, model: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Profile a model with the current configuration."""
        try:
            return await self._run_profiling(model, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Profiling failed: {str(e)}")
            raise

    async def _run_profiling(self, model: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Internal method to run the profiling process."""
        try:
            # Model profiling is handled by the @profile_model decorator
            result = await model(*args, **kwargs)
            return result
        except Exception as e:
            self.logger.error(f"Error during profiling execution: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Cleanup resources after profiling."""
        try:
            # Add any cleanup logic here
            self.logger.info("Cleanup completed successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

async def main() -> None:
    """Main entry point for the application."""
    try:
        memoraith = Memoraith()
        await memoraith.setup()
        # Example usage would go here
        await memoraith.cleanup()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
