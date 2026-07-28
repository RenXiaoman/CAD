#!/usr/bin/env python3

import torch
import torch.nn as nn

from .transunet3d_model import Generic_TransUNet_max_ppbp
from .vit_configs import CONFIGS_3D


def build_transunet3d_from_config(config) -> Generic_TransUNet_max_ppbp:
    return Generic_TransUNet_max_ppbp(
        input_channels=config.input_channels,
        base_num_features=config.base_num_features,
        num_classes=config.num_classes,
        num_pool=config.num_pool,
        num_conv_per_stage=config.num_conv_per_stage,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=config.deep_supervision,
        dropout_in_localization=config.dropout_in_localization,
        final_nonlin=lambda x: x,
        pool_op_kernel_sizes=config.pool_op_kernel_sizes,
        conv_kernel_sizes=config.conv_kernel_sizes,
        convolutional_pooling=config.convolutional_pooling,
        convolutional_upsampling=config.convolutional_upsampling,
        max_num_features=config.max_num_features,
        patch_size=config.patch_size,
        is_max=config.is_max,
        is_max_bottleneck_transformer=config.is_max_bottleneck_transformer,
        vit_depth=config.vit_depth,
        vit_hidden_size=config.vit_hidden_size,
        vit_mlp_dim=config.vit_mlp_dim,
        vit_num_heads=config.vit_num_heads,
    )


def main() -> None:
    config = CONFIGS_3D["R50-ViT-B_16"]()
    model = build_transunet3d_from_config(config).eval()
    x = torch.randn(1, config.input_channels, *config.patch_size)
    with torch.no_grad():
        output = model(x)

    outputs = list(output) if isinstance(output, tuple) else [output]
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Main output shape: {tuple(outputs[0].shape)}")
    print(f"All output shapes: {[tuple(item.shape) for item in outputs]}")
    print(f"Params: {total_params / 1e6:.2f} M")


if __name__ == "__main__":
    main()
