from pathlib import Path
import pickle

import numpy as np
import torch
from torch import nn

from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper


DEFAULT_PLANS = Path(__file__).with_name("nnUNetPlansv2.1_plans_3D.pkl")


def load_plans(plans_file=DEFAULT_PLANS):
    with Path(plans_file).open("rb") as f:
        return pickle.load(f)


def get_stage_plans(plans, stage=None):
    stages = plans["plans_per_stage"]
    if stage is None:
        if len(stages) != 1:
            raise ValueError(f"plans has {len(stages)} stages, please pass stage explicitly")
        stage = next(iter(stages))
    return stage, stages[stage]


def build_generic_unet_from_plans(plans_file=DEFAULT_PLANS, stage=None, deep_supervision=True):
    plans = load_plans(plans_file)
    stage, stage_plans = get_stage_plans(plans, stage)

    net_num_pool_op_kernel_sizes = stage_plans["pool_op_kernel_sizes"]
    net_conv_kernel_sizes = stage_plans["conv_kernel_sizes"]

    network = Generic_UNet(
        input_channels=plans["num_modalities"],
        base_num_features=plans["base_num_features"],
        num_classes=plans["num_classes"] + 1,
        num_pool=len(net_num_pool_op_kernel_sizes),
        num_conv_per_stage=plans["conv_per_stage"],
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv3d,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        deep_supervision=deep_supervision,
        dropout_in_localization=False,
        final_nonlin=lambda x: x,
        weightInitializer=InitWeights_He(1e-2),
        pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
        conv_kernel_sizes=net_conv_kernel_sizes,
        upscale_logits=False,
        convolutional_pooling=True,
        convolutional_upsampling=True,
    )
    network.inference_apply_nonlin = softmax_helper

    metadata = {
        "stage": stage,
        "patch_size": np.array(stage_plans["patch_size"]).astype(int).tolist(),
        "input_channels": plans["num_modalities"],
        "output_channels": plans["num_classes"] + 1,
        "pool_op_kernel_sizes": net_num_pool_op_kernel_sizes,
        "conv_kernel_sizes": net_conv_kernel_sizes,
        "batch_size_from_plans": stage_plans["batch_size"],
    }
    return network, metadata, plans


def set_inference_mode(network):
    network.eval()
    network.do_ds = False
    return network


def output_shapes(output):
    if torch.is_tensor(output):
        return [tuple(output.shape)]
    return [tuple(o.shape) for o in output]
