# ResGNet

Model code copied from https://github.com/MIAinCS/ResGNet.

Minimal usage:

```python
import torch
from models.ResGNet import ResGNet, VNet

model = VNet(ResGNet)
x = torch.randn(1, 1, 16, 256, 256)
y = model(x)
print(y.shape)  # [1, 1, 16, 256, 256]
```

Local changes:

- `__init__.py` only exports `ResGNet` and `VNet` to avoid optional training dependencies.
- `utils.py` uses a local default activation config (`elu`) instead of importing the original training `params.py`.
