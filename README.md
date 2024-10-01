# Memoraith

Memoraith is a cutting-edge, lightweight model profiler for deep learning frameworks, developed by Mehdi El Jouhfi. It's designed to revolutionize the optimization of neural network models by providing unparalleled insights into their performance characteristics.

## Features

- Advanced support for PyTorch and TensorFlow models
- High-precision profiling of memory usage (CPU and GPU)
- Microsecond-accurate computation time measurement for each layer
- Sophisticated bottleneck and anomaly detection algorithms
- Generation of comprehensive, interactive reports with advanced visualizations
- Real-time visualization capabilities with minimal overhead
- Flexible programmatic and command-line interfaces

## Installation

Install Memoraith using pip:

```bash
pip install memoraith
```

For GPU support and additional features:

```bash
pip install memoraith[full]
```

## Quick Start

Here's an example of Memoraith in action with a PyTorch model:

```python
from memoraith import profile_model, set_output_path
import torch
import torch.nn as nn

set_output_path('profiling_results/')

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

@profile_model(memory=True, computation=True, gpu=True, network=True)
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

This will generate a comprehensive profiling report in the 'profiling_results/' directory.

## Documentation

For detailed information on Memoraith's advanced features, please refer to our [comprehensive documentation](https://memoraith.readthedocs.io).

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

Memoraith is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Support

If you encounter any issues or have questions, please file an issue on the [GitHub issue tracker](https://github.com/mehdi342/Memoraith/issues).

## Citing Memoraith

If you use Memoraith in your research, please cite it as follows:

```bibtex
@software{memoraith,
  author = {El Jouhfi, Mehdi},
  title = {Memoraith: Advanced Lightweight Model Profiler for Deep Learning},
  year = {2024},
  url = {https://github.com/mehdi342/Memoraith}
}
```

## Contact

For inquiries, please contact Mehdi El Jouhfi at midojouhfi@gmail.com.