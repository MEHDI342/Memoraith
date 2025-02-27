<div align="center">
  <img src="assets/logo.png" alt="Memoraith Logo" width="600"/>

# Memoraith

[![PyPI version](https://badge.fury.io/py/memoraith.svg)](https://badge.fury.io/py/memoraith)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/memoraith/badge/?version=latest)](https://memoraith.readthedocs.io/en/latest/?badge=latest)

**Advanced Lightweight Model Profiler for Deep Learning**
</div>

## Overview

Memoraith is a cutting-edge, lightweight model profiler for deep learning frameworks, providing unparalleled insights into neural network performance. Developed with precision and efficiency in mind, it helps developers and researchers optimize their models through detailed performance analysis.

## ✨ Key Features

- 🔍 **Advanced Profiling**
    - High-precision memory tracking (CPU & GPU)
    - Microsecond-accurate computation timing
    - Layer-by-layer performance analysis

- 🎯 **Intelligent Analysis**
    - Sophisticated bottleneck detection
    - Anomaly identification
    - Optimization recommendations

- 📊 **Rich Visualization**
    - Interactive dashboards
    - Real-time monitoring
    - Comprehensive reports

- 🛠 **Framework Support**
    - PyTorch integration
    - TensorFlow support
    - Extensible architecture

## 🚀 Installation

Basic installation:
```bash
pip install memoraith
```

Full installation with GPU support and extra features:
```bash
pip install memoraith[full]
```

## 🎮 Quick Start

Here's a simple example using PyTorch:

```python
from memoraith import profile_model, set_output_path
import torch
import torch.nn as nn

# Set output directory for profiling results
set_output_path('profiling_results/')

# Define your model
class AdvancedNet(nn.Module):
    def __init__(self):
        super(AdvancedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Add profiling decorator
@profile_model(memory=True, computation=True, gpu=True)
def train_model(model):
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(100):
        input_data = torch.randn(32, 3, 32, 32)
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    model = AdvancedNet()
    train_model(model)
```

## 📚 Documentation

Visit our [comprehensive documentation](https://memoraith.readthedocs.io) for:
- Detailed API reference
- Advanced usage examples
- Best practices
- Troubleshooting guides

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:
- Code of conduct
- Development setup
- Submission guidelines
- Testing procedures

## 📝 License

Memoraith is released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🆘 Support

Need help?
- 📋 [GitHub Issues](https://github.com/mehdi342/Memoraith/issues)
- 📚 [Documentation](https://memoraith.readthedocs.io)
- 📧 [Email Support](mailto:midojouhfi@gmail.com)

## 📖 Citation

If you use Memoraith in your research, please cite:

```bibtex
@software{memoraith,
    author = {El Jouhfi, Mehdi},
    title = {Memoraith: Advanced Lightweight Model Profiler for Deep Learning},
    year = {2024},
    url = {https://github.com/mehdi342/Memoraith},
    version = {0.5.0}
}
```

## 📬 Contact

For inquiries, reach out to [Mehdi El Jouhfi](mailto:midojouhfi@gmail.com)

---
<div align="center">
  Made with ❤️ and sweat by Mehdi El Jouhfi
</div>
