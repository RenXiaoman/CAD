#!/usr/bin/env python3

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from monai.networks.nets import UNETR
from models.extracted_generic_unet.model_from_plans import build_generic_unet_from_plans
from picai_baseline.unet.training_setup.neural_networks.unets import UNet
from thop import profile


def count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    input_shape = (1, 1, 16, 256, 256)
    x = torch.randn(input_shape)
    nnunet_model, nnunet_meta, _ = build_generic_unet_from_plans(
        plans_file="/home/Space/clib/Projects/CAD/models/extracted_generic_unet/nnUNetPlansv2.1_plans_3D.pkl",
        deep_supervision=False,
    )

    models = {
        "UNet": UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
            channels=[32, 64, 128, 256, 512, 1024],
        ),
        "UNETR": UNETR(
            in_channels=1,
            out_channels=1,
            img_size=(16, 256, 256),
            spatial_dims=3,
        ),
        "Generic_UNet_from_plans": nnunet_model,
    }

    for name, model in models.items():
        model.eval()
        params = count_params(model)
        with torch.no_grad():
            output = model(x)
            flops, thop_params = profile(model, inputs=(x,), verbose=False)

        print(f"{name}")
        print(f"  Input shape:  {input_shape}")
        print(f"  Output shape: {tuple(output.shape)}")
        print(f"  Params (M):   {params / 1e6:.2f}")
        print(f"  FLOPs (G):    {flops / 1e9:.2f}")
        print(f"  THOP Params (M): {thop_params / 1e6:.2f}")
        if name == "Generic_UNet_from_plans":
            print(f"  Plans meta: {nnunet_meta}")


if __name__ == "__main__":
    main()
