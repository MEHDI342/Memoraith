import os
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import numpy as np
import tempfile
import time
from memoraith import profile_model, set_output_path

# Configure output directory for results
test_results_dir = tempfile.mkdtemp(prefix="memoraith_test_")
set_output_path(test_results_dir)
print(f"Test results directory: {test_results_dir}")

#==============================
# PyTorch Implementation
#==============================
class ResNetBlock(nn.Module):
    """ResNet basic block with residual connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SimpleResNet(nn.Module):
    """Simplified ResNet architecture for testing"""
    def __init__(self, num_blocks=2, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, num_blocks, stride=1)
        self.layer2 = self._make_layer(128, num_blocks, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

@profile_model(memory=True, computation=True, gpu=True, report_format='html')
def train_pytorch_model(model, epochs=3, batch_size=32):
    """Train a PyTorch model with synthetic data to test Memoraith profiling"""
    print("\n" + "="*50)
    print("PYTORCH MODEL PROFILING TEST")
    print("="*50)

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create synthetic dataset
    inputs = torch.randn(batch_size * 10, 3, 32, 32)
    targets = torch.randint(0, 10, (batch_size * 10,))

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0

        # Process batches
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/(len(inputs)/batch_size):.4f}")

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    return {
        "model_type": "PyTorch",
        "training_time": training_time,
        "epochs": epochs,
        "batch_size": batch_size
    }

#==============================
# TensorFlow Implementation
#==============================
def create_tf_model():
    """Create a TensorFlow model with similar architecture to the PyTorch model"""
    inputs = tf.keras.Input(shape=(32, 32, 3))

    # Initial convolution
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # First ResBlock group
    shortcut = x
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    # Second ResBlock with stride
    shortcut = tf.keras.layers.Conv2D(128, 1, strides=2, use_bias=False)(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    # Global pooling and final dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model

@profile_model(memory=True, computation=True, gpu=True, report_format='html')
def train_tensorflow_model(model, epochs=3, batch_size=32):
    """Train a TensorFlow model with synthetic data to test Memoraith profiling"""
    print("\n" + "="*50)
    print("TENSORFLOW MODEL PROFILING TEST")
    print("="*50)

    # Create synthetic dataset
    x_train = np.random.normal(size=(batch_size * 10, 32, 32, 3)).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(batch_size * 10)).astype(np.int32)

    # Enable memory growth for TensorFlow GPUs if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"TensorFlow using {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth configuration error: {e}")
    else:
        print("TensorFlow using CPU")

    # Train the model
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    return {
        "model_type": "TensorFlow",
        "training_time": training_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "final_loss": float(history.history["loss"][-1]),
        "final_accuracy": float(history.history["accuracy"][-1]) if "accuracy" in history.history else None
    }

#==============================
# Main Test Execution
#==============================
def run_comprehensive_test():
    """Run comprehensive tests for both frameworks"""
    results = {}

    # Test configuration
    batch_size = 32
    epochs = 2

    print("\n" + "="*80)
    print("MEMORAITH COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Configuration: {batch_size} batch size, {epochs} epochs")
    print(f"Results will be saved to: {test_results_dir}")

    try:
        # Test 1: PyTorch model profiling
        pt_model = SimpleResNet()
        pt_results = train_pytorch_model(pt_model, epochs=epochs, batch_size=batch_size)
        results["pytorch"] = pt_results

        # Test 2: TensorFlow model profiling
        tf_model = create_tf_model()
        tf_results = train_tensorflow_model(tf_model, epochs=epochs, batch_size=batch_size)
        results["tensorflow"] = tf_results

        # Display results summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        print(f"PyTorch training time: {results['pytorch']['training_time']:.2f} seconds")
        print(f"TensorFlow training time: {results['tensorflow']['training_time']:.2f} seconds")
        print("\nProfiling results location:")

        # List generated files
        for root, dirs, files in os.walk(test_results_dir):
            for file in files:
                print(f" - {os.path.join(root, file)}")

        return {
            "status": "success",
            "framework_results": results,
            "output_directory": test_results_dir
        }

    except Exception as e:
        print(f"ERROR: Test execution failed - {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "output_directory": test_results_dir
        }

if __name__ == "__main__":
    test_result = run_comprehensive_test()
    print(f"\nTest completed with status: {test_result['status']}")