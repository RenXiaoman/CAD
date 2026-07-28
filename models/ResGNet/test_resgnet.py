#!/usr/bin/env python3

import torch
from thop import profile
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.ResGNet import ResGNet, VNet


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (1, 1, 16, 256, 256)
    model = VNet(ResGNet).to(device).eval()
    x = torch.randn(input_shape).to(device)

    with torch.no_grad():
        output = model(x)
        flops, thop_params = profile(model, inputs=(x,), verbose=False)

    params = sum(param.numel() for param in model.parameters())

    print(f"Input shape:  {input_shape}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Params:       {params / 1e6:.2f} M")
    print(f"FLOPs:        {flops / 1e9:.2f} G")
    print(f"THOP Params:  {thop_params / 1e6:.2f} M")


if __name__ == "__main__":
    main()
