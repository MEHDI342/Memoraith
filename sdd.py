#!/usr/bin/env python
# memoraith_verification.py - Production Readiness Test
# Usage: python memoraith_verification.py

import os
import time
import tempfile
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memoraith-test")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}\n")

def test_imports():
    """Test importing all critical Memoraith components."""
    print_section("TESTING MEMORAITH IMPORTS")

    try:
        # Import main package components
        import memoraith
        print(f"✓ Main package imported successfully (version: {memoraith.__version__})")

        # Import core profiling functionality
        from memoraith import profile_model, set_output_path
        print("✓ Profiling API imported successfully")

        # Import configuration
        from memoraith.config import Config, config
        print("✓ Configuration module imported successfully")

        # Import exceptions
        from memoraith.exceptions import MemoraithError
        print("✓ Exception handling imported successfully")

        # Test framework adapters
        from memoraith.integration import get_framework_adapter
        from memoraith.integration.common_utils import identify_framework
        print("✓ Framework integration modules imported successfully")

        # Test analysis components
        from memoraith.analysis import Analyzer
        print("✓ Analysis modules imported successfully")

        # Test reporting components
        from memoraith.reporting import ReportGenerator
        print("✓ Reporting modules imported successfully")

        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"✗ Failed to import required module: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during import testing: {e}")
        print(f"✗ Unexpected error: {e}")
        return False

def test_pytorch_profiling():
    """Test PyTorch model profiling functionality."""
    print_section("TESTING PYTORCH PROFILING")

    try:
        import torch
        import torch.nn as nn
        from memoraith import profile_model, set_output_path

        # Create a temporary directory for profiling output
        output_dir = tempfile.mkdtemp(prefix="memoraith_pytorch_test_")
        set_output_path(output_dir)
        print(f"✓ Output directory set to: {output_dir}")

        # Define a simple CNN model for testing
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.relu = nn.ReLU(inplace=True)
                self.fc = nn.Linear(16 * 32 * 32, 10)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        # Create the model and a sample input
        model = SimpleCNN()
        input_data = torch.randn(2, 3, 32, 32)
        print("✓ Test model created successfully")

        # Define a function to profile
        @profile_model(memory=True, computation=True, gpu=False, save_report=True)
        def run_inference(model, data):
            print("  - Running inference inside profiled function")
            start_time = time.time()
            output = model(data)
            elapsed = time.time() - start_time
            return {
                "output_shape": tuple(output.shape),
                "elapsed_time": elapsed
            }

        # Run the profiled function
        print("- Running profiled inference...")
        result = run_inference(model, input_data)
        print(f"✓ Profiled function executed successfully")
        print(f"  - Output shape: {result['output_shape']}")
        print(f"  - Raw inference time: {result['elapsed_time']:.4f}s")

        # Verify output files were created
        output_files = list(Path(output_dir).glob("**/*"))
        report_files = [f for f in output_files if f.name.endswith(('.html', '.json', '.png'))]

        if report_files:
            print(f"✓ Generated {len(report_files)} profiling artifacts:")
            for file_path in report_files[:5]:  # Show only first 5 to avoid clutter
                print(f"  - {file_path.name}")
            if len(report_files) > 5:
                print(f"  - ... and {len(report_files) - 5} more files")
            return True
        else:
            print("✗ No profiling output files were generated")
            return False

    except ImportError as e:
        print(f"✗ PyTorch test skipped: {e}")
        logger.info(f"PyTorch not available: {e}")
        return None  # Skip but don't consider a failure
    except Exception as e:
        logger.error(f"PyTorch profiling test failed: {e}", exc_info=True)
        print(f"✗ PyTorch profiling test failed: {e}")
        return False

def test_tensorflow_profiling():
    """Test TensorFlow model profiling functionality."""
    print_section("TESTING TENSORFLOW PROFILING")

    try:
        import tensorflow as tf
        from memoraith import profile_model, set_output_path

        # Create a temporary directory for profiling output
        output_dir = tempfile.mkdtemp(prefix="memoraith_tensorflow_test_")
        set_output_path(output_dir)
        print(f"✓ Output directory set to: {output_dir}")

        # Define a simple TensorFlow model
        def create_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        # Create model and sample data
        model = create_model()
        input_data = tf.random.normal((4, 10))
        print("✓ Test model created successfully")

        # Define a function to profile
        @profile_model(memory=True, computation=True, gpu=False, save_report=True)
        def run_inference(model, data):
            print("  - Running inference inside profiled function")
            start_time = time.time()
            output = model(data)
            elapsed = time.time() - start_time
            return {
                "output_shape": tuple(output.shape),
                "elapsed_time": elapsed
            }

        # Run the profiled function
        print("- Running profiled inference...")
        result = run_inference(model, input_data)
        print(f"✓ Profiled function executed successfully")
        print(f"  - Output shape: {result['output_shape']}")
        print(f"  - Raw inference time: {result['elapsed_time']:.4f}s")

        # Verify output files were created
        output_files = list(Path(output_dir).glob("**/*"))
        report_files = [f for f in output_files if f.name.endswith(('.html', '.json', '.png'))]

        if report_files:
            print(f"✓ Generated {len(report_files)} profiling artifacts:")
            for file_path in report_files[:5]:  # Show only first 5 to avoid clutter
                print(f"  - {file_path.name}")
            if len(report_files) > 5:
                print(f"  - ... and {len(report_files) - 5} more files")
            return True
        else:
            print("✗ No profiling output files were generated")
            return False

    except ImportError as e:
        print(f"✗ TensorFlow test skipped: {e}")
        logger.info(f"TensorFlow not available: {e}")
        return None  # Skip but don't consider a failure
    except Exception as e:
        logger.error(f"TensorFlow profiling test failed: {e}", exc_info=True)
        print(f"✗ TensorFlow profiling test failed: {e}")
        return False

def test_system_checks():
    """Perform system compatibility checks."""
    print_section("SYSTEM COMPATIBILITY CHECKS")

    try:
        import psutil
        import os
        import platform

        # System information
        print(f"Operating System: {platform.system()} {platform.version()}")
        print(f"Python Version: {platform.python_version()}")

        # CPU information
        cpu_count = os.cpu_count()
        print(f"CPU Cores: {cpu_count}")

        # Memory information
        memory = psutil.virtual_memory()
        print(f"Available Memory: {memory.available / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB")

        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                print(f"CUDA Available: Yes (Device count: {torch.cuda.device_count()})")
                for i in range(torch.cuda.device_count()):
                    print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                    props = torch.cuda.get_device_properties(i)
                    print(f"    ├── Memory: {props.total_memory / (1024**3):.2f} GB")
                    print(f"    └── CUDA Capability: {props.major}.{props.minor}")
            else:
                print("CUDA Available: No")
        except (ImportError, AttributeError):
            print("CUDA Status: Unknown (PyTorch not available)")

        # TensorFlow GPU information
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"TensorFlow GPU(s): {len(gpus)}")
                for gpu in gpus:
                    print(f"  - {gpu.name}")
            else:
                print("TensorFlow GPU(s): None")
        except ImportError:
            print("TensorFlow Status: Not available")

        return True
    except Exception as e:
        print(f"✗ System check error: {e}")
        return False

def run_all_tests():
    """Run all verification tests and summarize results."""
    print_section("MEMORAITH VERIFICATION TEST")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    results = {}

    # Run import tests first
    results["imports"] = test_imports()

    # If imports succeed, proceed with functional tests
    if results["imports"]:
        results["system"] = test_system_checks()
        results["pytorch"] = test_pytorch_profiling()
        results["tensorflow"] = test_tensorflow_profiling()

    # Summarize results
    print_section("TEST RESULTS SUMMARY")
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"{test_name.capitalize():12} : {status}")

    # Overall status
    # Ignore skipped tests (None values) when determining success
    actual_tests = [r for r in results.values() if r is not None]
    if all(actual_tests):
        print("\n✅ VERIFICATION SUCCESSFUL: Memoraith is working correctly!")
        return 0
    else:
        print("\n❌ VERIFICATION FAILED: See above errors for details.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())