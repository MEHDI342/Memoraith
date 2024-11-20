# config.py
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import yaml
import json
import torch.optim as optim
import torch.nn as nn

class Config:
    """Finnaly a good config ."""

    def __init__(self):
        # Core settings
        self.output_path = Path('memoraith_reports/')
        self.enable_gpu = False
        self.enable_memory = True
        self.enable_time = True
        self.report_format = 'html'
        self.real_time_viz = False

        # Profiling settings
        self.profiling_interval = 0.1
        self.max_memory_samples = 1000
        self.bottleneck_threshold = 0.1
        self.anomaly_threshold = 3.0

        # Training settings
        self.batch_size = 32
        self.max_epochs = 100
        self.learning_rate = 0.001
        self.optimizer = 'adam'
        self.loss_function = 'cross_entropy'

        # Logging settings
        self.log_level = logging.INFO
        self.enable_console_output = True
        self.enable_file_logging = True

        # Load environment variables
        load_dotenv()
        self.load_from_env()

    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        self.output_path = Path(os.getenv('MEMORAITH_OUTPUT_PATH', str(self.output_path)))
        self.enable_gpu = os.getenv('MEMORAITH_ENABLE_GPU', str(self.enable_gpu)).lower() == 'true'
        self.enable_memory = os.getenv('MEMORAITH_ENABLE_MEMORY', str(self.enable_memory)).lower() == 'true'
        self.enable_time = os.getenv('MEMORAITH_ENABLE_TIME', str(self.enable_time)).lower() == 'true'
        self.batch_size = int(os.getenv('MEMORAITH_BATCH_SIZE', str(self.batch_size)))
        self.max_epochs = int(os.getenv('MEMORAITH_MAX_EPOCHS', str(self.max_epochs)))
        self.learning_rate = float(os.getenv('MEMORAITH_LEARNING_RATE', str(self.learning_rate)))
        self.optimizer = os.getenv('MEMORAITH_OPTIMIZER', self.optimizer)
        self.loss_function = os.getenv('MEMORAITH_LOSS_FUNCTION', self.loss_function)

    def load_from_file(self, config_file: str) -> None:
        """Load configuration from a YAML file."""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save_to_file(self, filename: str) -> None:
        """Save the current configuration to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Configuration saved to {filename}")

    def get_optimizer(self, parameters: Any) -> Optional[Any]:
        """Get the optimizer instance based on the configuration."""
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        optimizer_class = optimizer_map.get(self.optimizer.lower())
        return optimizer_class(parameters, lr=self.learning_rate) if optimizer_class else None

    def get_loss_function(self) -> Optional[Any]:
        """Get the loss function based on the configuration."""
        loss_map = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'bce': nn.BCELoss
        }
        loss_class = loss_map.get(self.loss_function.lower())
        return loss_class() if loss_class else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {k: str(v) if isinstance(v, Path) else v
                for k, v in self.__dict__.items()
                if not k.startswith('_')}

    def validate(self) -> bool:
        """Validate the current configuration."""
        valid = True
        if not isinstance(self.output_path, Path):
            logging.error("output_path must be a Path object")
            valid = False
        if not isinstance(self.enable_gpu, bool):
            logging.error("enable_gpu must be a boolean")
            valid = False
        return valid

# Global configuration instance
config = Config()