# Extracted nnUNet Generic_UNet

This directory extracts the nnUNet v1 `Generic_UNet` model construction from the
trainer so that another training or inference pipeline can instantiate the same
network directly.

The expected input tensor shape is:

```text
[B, C, D, H, W]
```

For the included Task130 plans, the default patch input is:

```text
[B, 1, 16, 256, 256]
```

## Files

```text
extracted_generic_unet/
├── nnUNetPlansv2.1_plans_3D.pkl
├── model_from_plans.py
└── test_forward.py
```

`nnUNetPlansv2.1_plans_3D.pkl` is copied from:

```text
workdir/nnUNet_preprocessed/Task130_ProstateAHCDU/nnUNetPlansv2.1_plans_3D.pkl
```

It stores the architecture decisions made by nnUNet planning, including input
modalities, number of classes, patch size, pooling kernels, convolution kernels,
batch size, spacing, and stage configuration.

`model_from_plans.py` is the reusable part. It reads the pkl and builds the same
`Generic_UNet` that `nnUNetTrainerV2` would build for `3d_fullres`.

`test_forward.py` is only a smoke test. It creates a random tensor and checks
that the model can run forward.

## What Was Extracted

In nnUNet v1, `nnUNetTrainerV2.initialize_network()` builds the model like this:

```python
Generic_UNet(
    num_input_channels,
    base_num_features,
    num_classes,
    len(pool_op_kernel_sizes),
    conv_per_stage,
    2,
    nn.Conv3d,
    nn.InstanceNorm3d,
    {"eps": 1e-5, "affine": True},
    nn.Dropout3d,
    {"p": 0, "inplace": True},
    nn.LeakyReLU,
    {"negative_slope": 1e-2, "inplace": True},
    deep_supervision=True,
    dropout_in_localization=False,
    final_nonlin=lambda x: x,
    weightInitializer=InitWeights_He(1e-2),
    pool_op_kernel_sizes=...,
    conv_kernel_sizes=...,
    upscale_logits=False,
    convolutional_pooling=True,
    convolutional_upsampling=True,
)
```

`model_from_plans.py` recreates this setup without creating a trainer.

## Important pkl Fields

These fields are read from the plans file:

```text
plans["num_modalities"]       -> input channels C
plans["num_classes"] + 1      -> output channels, including background
plans["base_num_features"]    -> first feature width
plans["conv_per_stage"]       -> number of convs per encoder/decoder stage
stage["patch_size"]           -> recommended [D, H, W]
stage["pool_op_kernel_sizes"] -> downsampling schedule
stage["conv_kernel_sizes"]    -> convolution kernel schedule
```

For the current Task130 pkl:

```text
input_channels: 1
output_channels: 2
patch_size: [16, 256, 256]
pool_op_kernel_sizes:
  [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2], [1, 2, 2]]
conv_kernel_sizes:
  [[1, 3, 3], [1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
```

The output channel count is `2` because nnUNet stores foreground classes in
`plans["num_classes"]`, then adds background when building the network.

## Basic Use

From the repo root:

```bash
/home/ikun_server/anaconda3/envs/nnUNetv1/bin/python extracted_generic_unet/test_forward.py --device cuda:0
```

Use a custom input shape:

```bash
/home/ikun_server/anaconda3/envs/nnUNetv1/bin/python extracted_generic_unet/test_forward.py \
  --device cuda:0 \
  --shape "1 1 16 256 256"
```

The shape format is:

```text
B C D H W
```

## Reuse in Another Model or Training Script

Import the builder:

```python
from pathlib import Path
import torch

from extracted_generic_unet.model_from_plans import (
    build_generic_unet_from_plans,
    set_inference_mode,
)

net, meta, plans = build_generic_unet_from_plans(
    plans_file=Path("extracted_generic_unet/nnUNetPlansv2.1_plans_3D.pkl"),
    deep_supervision=False,
)

set_inference_mode(net)
net = net.to("cuda:0")

x = torch.randn(1, meta["input_channels"], *meta["patch_size"], device="cuda:0")
y = net(x)

print(y.shape)
```

Expected output shape for Task130:

```text
[1, 2, 16, 256, 256]
```

## Deep Supervision

By default, nnUNetTrainerV2 trains with deep supervision enabled. In that mode,
the network returns multiple outputs at different scales.

For normal inference or for plugging the model into a simpler external pipeline,
use:

```python
net, meta, plans = build_generic_unet_from_plans(deep_supervision=False)
set_inference_mode(net)
```

Then `net(x)` returns one full-resolution tensor.

If you pass `deep_supervision=True`, `net(x)` may return a tuple/list of outputs.
The helper `output_shapes(output)` in `model_from_plans.py` can print all output
shapes.

## Notes

This does not copy nnUNet's trainer, optimizer, data augmentation, loss, sliding
window inference, mirroring, postprocessing, or preprocessing code. It only
extracts the architecture creation step.

The model still imports `Generic_UNet` from the local nnUNet codebase:

```python
from nnunet.network_architecture.generic_UNet import Generic_UNet
```

So run it from this repository, or ensure this repository is on `PYTHONPATH`.
