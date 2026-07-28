#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


def _load_transunet3d_package():
    package_name = "transunet3d"
    if package_name in sys.modules:
        return

    package_dir = Path(__file__).resolve().parent / "3D-TransUNet"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load TransUNet3D package from {package_dir}")

    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)


class MainOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


def build_transunet3d_main_output(config_name: str = "R50-ViT-B_16") -> nn.Module:
    _load_transunet3d_package()

    from transunet3d.test_forward_3d import build_transunet3d_from_config
    from transunet3d.vit_configs import CONFIGS_3D

    config = CONFIGS_3D[config_name]()
    return MainOutputWrapper(build_transunet3d_from_config(config))
