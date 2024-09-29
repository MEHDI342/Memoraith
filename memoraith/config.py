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
    """
    Comprehensive configuration management for Memoraith.
    Includes all existing functionality plus enhancements.
    """

    def __init__(self):
        self.output_path = Path('memoraith_reports/')
        self.enable_gpu = False
        self.enable_memory = True
        self.enable_time = True
        self.log_level = logging.INFO
        self.report_format = 'html'
        self.real_time_viz = False
        self.profiling_interval = 0.1
        self.max_memory_samples = 1000
        self.bottleneck_threshold = 0.1
        self.anomaly_threshold = 3.0
        self.batch_size = 32
        self.max_epochs = 100
        self.learning_rate = 0.001
        self.optimizer = 'adam'
        self.loss_function = 'cross_entropy'

        # Load environment variables
        load_dotenv()

        # Load config from environment variables
        self.load_from_env()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def set_output_path(self, path: str) -> None:
        """Set the output path for profiling reports."""
        self.output_path = Path(path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def enable_gpu_profiling(self, enable: bool) -> None:
        """Enable or disable GPU profiling."""
        self.enable_gpu = enable

    def set_log_level(self, level: int) -> None:
        """Set the logging level."""
        self.log_level = level

    def set_batch_size(self, size: int) -> None:
        """Set the batch size for training."""
        self.batch_size = size

    def set_max_epochs(self, epochs: int) -> None:
        """Set the maximum number of epochs for training."""
        self.max_epochs = epochs

    def set_learning_rate(self, lr: float) -> None:
        """Set the learning rate for training."""
        self.learning_rate = lr

    def set_optimizer(self, optimizer: str) -> None:
        """Set the optimizer for training."""
        self.optimizer = optimizer

    def set_loss_function(self, loss: str) -> None:
        """Set the loss function for training."""
        self.loss_function = loss

    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        self.output_path = Path(os.getenv('MEMORAITH_OUTPUT_PATH', str(self.output_path)))
        self.enable_gpu = os.getenv('MEMORAITH_ENABLE_GPU', str(self.enable_gpu)).lower() == 'true'
        self.enable_memory = os.getenv('MEMORAITH_ENABLE_MEMORY', str(self.enable_memory)).lower() == 'true'
        self.enable_time = os.getenv('MEMORAITH_ENABLE_TIME', str(self.enable_time)).lower() == 'true'
        self.log_level = getattr(logging, os.getenv('MEMORAITH_LOG_LEVEL', 'INFO'))
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

        # Update main attributes if they're in the loaded config
        if 'output_path' in config_data:
            self.set_output_path(config_data['output_path'])
        if 'enable_gpu' in config_data:
            self.enable_gpu_profiling(config_data['enable_gpu'])
        if 'log_level' in config_data:
            self.set_log_level(getattr(logging, config_data['log_level']))
        if 'batch_size' in config_data:
            self.set_batch_size(config_data['batch_size'])
        if 'max_epochs' in config_data:
            self.set_max_epochs(config_data['max_epochs'])
        if 'learning_rate' in config_data:
            self.set_learning_rate(config_data['learning_rate'])
        if 'optimizer' in config_data:
            self.set_optimizer(config_data['optimizer'])
        if 'loss_function' in config_data:
            self.set_loss_function(config_data['loss_function'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def get_optimizer(self, parameters: Any) -> Optional[Any]:
        """Get the optimizer instance based on the configuration."""
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            # Add more optimizers as needed
        }
        optimizer_class = optimizer_map.get(self.optimizer.lower())
        if optimizer_class:
            return optimizer_class(parameters, lr=self.learning_rate)
        else:
            logging.error(f"Optimizer '{self.optimizer}' not supported.")
            return None

    def get_loss_function(self) -> Optional[Any]:
        """Get the loss function based on the configuration."""
        loss_map = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'bce': nn.BCELoss,
            # Add more loss functions as needed
        }
        loss_class = loss_map.get(self.loss_function.lower())
        if loss_class:
            return loss_class()
        else:
            logging.error(f"Loss function '{self.loss_function}' not supported.")
            return None

    def save_to_file(self, filename: str) -> None:
        """Save the current configuration to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Configuration saved to {filename}")

    def validate(self) -> bool:
        """Validate the current configuration."""
        # Add validation logic here
        valid = True
        if not isinstance(self.output_path, Path):
            logging.error("output_path must be a Path object")
            valid = False
        if not isinstance(self.enable_gpu, bool):
            logging.error("enable_gpu must be a boolean")
            valid = False
        # Add more validation checks as needed
        return valid

    def set_profiling_interval(self, interval: float) -> None:
        """Set the profiling interval."""
        self.profiling_interval = interval

    def set_max_memory_samples(self, samples: int) -> None:
        """Set the maximum number of memory samples to collect."""
        self.max_memory_samples = samples

    def set_bottleneck_threshold(self, threshold: float) -> None:
        """Set the threshold for detecting bottlenecks."""
        self.bottleneck_threshold = threshold

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Set the threshold for detecting anomalies."""
        self.anomaly_threshold = threshold

    def enable_real_time_visualization(self, enable: bool) -> None:
        """Enable or disable real-time visualization."""
        self.real_time_viz = enable

    def set_report_format(self, format: str) -> None:
        """Set the report format (html or pdf)."""
        if format.lower() in ['html', 'pdf']:
            self.report_format = format.lower()
        else:
            logging.error(f"Unsupported report format: {format}. Using default (html).")

    def get_full_config(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary."""
        return self.to_dict()

    def reset_to_defaults(self) -> None:
        """Reset all configuration options to their default values."""
        self.__init__()

    def __str__(self) -> str:
        """String representation of the Config object."""
        return f"Config(output_path={self.output_path}, enable_gpu={self.enable_gpu}, ...)"

    def __repr__(self) -> str:
        """Detailed string representation of the Config object."""
        return self.__str__()

# Global configuration instance
config = Config()
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
    """
    Comprehensive configuration management for Memoraith.
    Includes all existing functionality plus enhancements.
    """

    def __init__(self):
        self.output_path = Path('memoraith_reports/')
        self.enable_gpu = False
        self.enable_memory = True
        self.enable_time = True
        self.log_level = logging.INFO
        self.report_format = 'html'
        self.real_time_viz = False
        self.profiling_interval = 0.1
        self.max_memory_samples = 1000
        self.bottleneck_threshold = 0.1
        self.anomaly_threshold = 3.0
        self.batch_size = 32
        self.max_epochs = 100
        self.learning_rate = 0.001
        self.optimizer = 'adam'
        self.loss_function = 'cross_entropy'

        # Load environment variables
        load_dotenv()

        # Load config from environment variables
        self.load_from_env()

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def set_output_path(self, path: str) -> None:
        """Set the output path for profiling reports."""
        self.output_path = Path(path)
        self.output_path.mkdir(parents=True, exist_ok=True)

    def enable_gpu_profiling(self, enable: bool) -> None:
        """Enable or disable GPU profiling."""
        self.enable_gpu = enable

    def set_log_level(self, level: str) -> None:
        """Set the logging level."""
        self.log_level = level

    def set_batch_size(self, size: int) -> None:
        """Set the batch size for training."""
        self.batch_size = size

    def set_max_epochs(self, epochs: int) -> None:
        """Set the maximum number of epochs for training."""
        self.max_epochs = epochs

    def set_learning_rate(self, lr: float) -> None:
        """Set the learning rate for training."""
        self.learning_rate = lr

    def set_optimizer(self, optimizer: str) -> None:
        """Set the optimizer for training."""
        self.optimizer = optimizer

    def set_loss_function(self, loss: str) -> None:
        """Set the loss function for training."""
        self.loss_function = loss

    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        self.output_path = Path(os.getenv('MEMORAITH_OUTPUT_PATH', str(self.output_path)))
        self.enable_gpu = os.getenv('MEMORAITH_ENABLE_GPU', str(self.enable_gpu)).lower() == 'true'
        self.enable_memory = os.getenv('MEMORAITH_ENABLE_MEMORY', str(self.enable_memory)).lower() == 'true'
        self.enable_time = os.getenv('MEMORAITH_ENABLE_TIME', str(self.enable_time)).lower() == 'true'
        self.log_level = os.getenv('MEMORAITH_LOG_LEVEL', 'INFO')
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

        # Update main attributes if they're in the loaded config
        if 'output_path' in config_data:
            self.set_output_path(config_data['output_path'])
        if 'enable_gpu' in config_data:
            self.enable_gpu_profiling(config_data['enable_gpu'])
        if 'log_level' in config_data:
            self.log_level = config_data['log_level']
        if 'batch_size' in config_data:
            self.set_batch_size(config_data['batch_size'])
        if 'max_epochs' in config_data:
            self.set_max_epochs(config_data['max_epochs'])
        if 'learning_rate' in config_data:
            self.set_learning_rate(config_data['learning_rate'])
        if 'optimizer' in config_data:
            self.set_optimizer(config_data['optimizer'])
        if 'loss_function' in config_data:
            self.set_loss_function(config_data['loss_function'])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def get_optimizer(self, parameters: Any) -> Optional[Any]:
        """Get the optimizer instance based on the configuration."""
        optimizer_map = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
            # Add more optimizers as needed
        }
        optimizer_class = optimizer_map.get(self.optimizer.lower())
        if optimizer_class:
            return optimizer_class(parameters, lr=self.learning_rate)
        else:
            logging.error(f"Optimizer '{self.optimizer}' not supported.")
            return None

    def get_loss_function(self) -> Optional[Any]:
        """Get the loss function based on the configuration."""
        loss_map = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'bce': nn.BCELoss,
            # Add more loss functions as needed
        }
        loss_class = loss_map.get(self.loss_function.lower())
        if loss_class:
            return loss_class()
        else:
            logging.error(f"Loss function '{self.loss_function}' not supported.")
            return None

    def save_to_file(self, filename: str) -> None:
        """Save the current configuration to a JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logging.info(f"Configuration saved to {filename}")

    def validate(self) -> bool:
        """Validate the current configuration."""
        # Add validation logic here
        valid = True
        if not isinstance(self.output_path, Path):
            logging.error("output_path must be a Path object")
            valid = False
        if not isinstance(self.enable_gpu, bool):
            logging.error("enable_gpu must be a boolean")
            valid = False
        # Add more validation checks as needed
        return valid

    def set_profiling_interval(self, interval: float) -> None:
        """Set the profiling interval."""
        self.profiling_interval = interval

    def set_max_memory_samples(self, samples: int) -> None:
        """Set the maximum number of memory samples to collect."""
        self.max_memory_samples = samples

    def set_bottleneck_threshold(self, threshold: float) -> None:
        """Set the threshold for detecting bottlenecks."""
        self.bottleneck_threshold = threshold

    def set_anomaly_threshold(self, threshold: float) -> None:
        """Set the threshold for detecting anomalies."""
        self.anomaly_threshold = threshold

    def enable_real_time_visualization(self, enable: bool) -> None:
        """Enable or disable real-time visualization."""
        self.real_time_viz = enable

    def set_report_format(self, format: str) -> None:
        """Set the report format (html or pdf)."""
        if format.lower() in ['html', 'pdf']:
            self.report_format = format.lower()
        else:
            logging.error(f"Unsupported report format: {format}. Using default (html).")

    def get_full_config(self) -> Dict[str, Any]:
        """Get the full configuration as a dictionary."""
        return self.to_dict()

    def reset_to_defaults(self) -> None:
        """Reset all configuration options to their default values."""
        self.__init__()

    def __str__(self) -> str:
        """String representation of the Config object."""
        return f"Config(output_path={self.output_path}, enable_gpu={self.enable_gpu}, ...)"

    def __repr__(self) -> str:
        """Detailed string representation of the Config object."""
        return self.__str__()

# Global configuration instance
config = Config()