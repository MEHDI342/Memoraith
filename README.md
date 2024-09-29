# Memoraith

Memoraith is a lightweight model profiler for deep learning frameworks, designed to help you optimize your neural network models by providing detailed insights into their performance characteristics.

## Features

- Supports PyTorch and TensorFlow models
- Profiles memory usage (CPU and GPU)
- Measures computation time for each layer
- Detects bottlenecks and anomalies
- Generates comprehensive reports with visualizations
- Provides real-time visualization capabilities
- Offers both programmatic and command-line interfaces

## Installation

You can install Memoraith using pip:

```bash
pip install memoraith
```

For GPU support, install with:

```bash
pip install memoraith[gpu]
```

## Quick Start

Here's a simple example of how to use Memoraith with a PyTorch model:

```python
from memoraith import profile_model, set_output_path
import torch
import torch.nn as nn

set_output_path('profiling_results/')

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

@profile_model(memory=True, computation=True, gpu=True)
def train_model(model):
    optimizer = torch.optim.Adam(model.parameters())
    for _ in range(100):
        input_data = torch.randn(32, 10)
        output = model(input_data)
        loss = output.sum()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    model = SimpleNet()
    train_model(model)
```

This will generate a profiling report in the 'profiling_results/' directory.

## Documentation

For more detailed information on how to use Memoraith, please refer to our [documentation](https://memoraith.readthedocs.io).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

Memoraith is released under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Support

If you encounter any issues or have questions, please file an issue on the [GitHub issue tracker](https://github.com/yourusername/memoraith/issues).

## Citing Memoraith

If you use Memoraith in your research, please cite it as follows:

```
@software{memoraith,
  author = {Your Name},
  title = {Memoraith: A Lightweight Model Profiler for Deep Learning},
  year = {2023},
  url = {https://github.com/yourusername/memoraith}
}
```