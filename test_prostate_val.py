#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import subprocess
import sys
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree
from thop import profile

#### Model Lists ####
from picai_baseline.unet.training_setup.neural_networks.unets import UNet
from monai.networks.nets import UNETR
from monai.networks.nets import AttentionUnet
from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.convolutions import ResidualUnit
from monai.networks.layers.factories import Norm
from models.DIY.diy import DIY
from models.DIY.DIYgate import DIYGate
from models.DIY.DIY_boundary import DIYBoundary
from models.DIY.DIY_boundary_msag import DIYBoundaryMSAG
from models.DIY.ablation import BoundaryOnlyNet, MSAGOnlyNet

from models.extracted_generic_unet.model_from_plans import build_generic_unet_from_plans

from models.network_architecture.BMANet import BMANet
from models.WaveFormer.network_backbone import Waveformer
from models.transunet3d_builder import build_transunet3d_main_output

from models.ResGNet import ResGNet, VNet  # single dimension output单通道输出

from models.awa_boundary_builder import build_aw_unet, build_boundary_dou_loss
from models.WaTER import UNet3D as WaTER_UNet3D


class DIYNoWave(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: int = 3,
        feature_size: int = 16,
        norm_name: tuple | str = "instance",
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.stem = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.enc1 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.enc2 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=4 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.enc3 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=8 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.enc4 = ResidualUnit(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=16 * feature_size,
            strides=2,
            kernel_size=3,
            adn_ordering="NDA",
            act=("leakyrelu", {"negative_slope": 0.2}),
            norm=Norm.BATCH,
            dropout=dropout_rate,
            dropout_dim=1,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        up4 = self.decoder5(x5, x4)
        up3 = self.decoder4(up4, x3)
        up2 = self.decoder3(up3, x2)
        up1 = self.decoder2(up2, x1)
        return self.out(up1)

# Model
# from picai_baseline.unet.training_setup.neural_networks.unets import UNet

# Local imports
from gland_dataset_nnunet import Lits_DataSet

parser = argparse.ArgumentParser(description='Test prostate validation set')
parser.add_argument('--model_path', type=str,default='checkpoints/SegGland_WaTER_PICAI_NoAug/best_dice_model.pth', required=False, help='Path to trained model checkpoint')

parser.add_argument('--data_path', type=str, default='dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI', help='Path to dataset')
# parser.add_argument('--data_path', type=str, default='dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU', help='Path to dataset')

parser.add_argument('--output_dir', type=str, default='val_results', help='Output directory for predictions')
parser.add_argument('--slices_dir', type=str, default='slice_contours', help='Directory for slice images with contours')
parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
parser.add_argument('--plot_workers', type=int, default=8, help='Number of worker processes used to generate contour images')
parser.add_argument('--skip_plots', action='store_true', help='Stop after metrics, JSON, and NIfTI predictions are saved')
parser.add_argument('--keep_largest_cc', action='store_true',default=False, help='Keep only the largest connected component in predictions')
parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'], help='Mode: val for validation, test for testing')

args = parser.parse_args()
    
    
def calculate_dice(preds, targets):
    """Calculate Dice coefficient"""
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    pred_sum = preds_flat.sum()
    target_sum = targets_flat.sum()
    
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.item()

def calculate_miou(preds, targets):
    """Calculate mean Intersection over Union"""
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()

def calculate_sensitivity_precision(preds, targets):
    """Calculate sensitivity and precision for foreground segmentation."""
    smooth = 1e-6
    preds_flat = preds.view(-1).bool()
    targets_flat = targets.view(-1).bool()

    tp = (preds_flat & targets_flat).sum().float()
    fp = (preds_flat & ~targets_flat).sum().float()
    fn = (~preds_flat & targets_flat).sum().float()

    sensitivity = (tp + smooth) / (tp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    return sensitivity.item(), precision.item()

def calculate_model_complexity(model, device, input_shape=(1, 1, 16, 256, 256)):
    """Calculate Params (M) and FLOPs (G) with a fixed dummy input."""
    was_training = model.training
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)

    with torch.no_grad():
        flops, thop_params = profile(model, inputs=(dummy_input,), verbose=False)

    if was_training:
        model.train()

    params = sum(p.numel() for p in model.parameters())
    return {
        "input_shape": list(input_shape),
        "params_m": round(params / 1e6, 4),
        "thop_params_m": round(thop_params / 1e6, 4),
        "flops_g": round(flops / 1e9, 4),
    }

def format_mean_std(mean_value, std_value, scale=1.0):
    return f"{mean_value * scale:.2f} ± {std_value * scale:.2f}"

def get_surface_points(binary_mask):
    """
    Get surface points of a binary mask

    Args:
        binary_mask: Binary mask (numpy array)

    Returns:
        Array of surface point coordinates
    """
    # Create a structuring element for erosion (26-connectivity for 3D)
    struct = generate_binary_structure(3, 3)

    # Erode the mask
    eroded = binary_erosion(binary_mask, struct)

    # Surface = original mask - eroded mask
    surface = binary_mask.astype(np.uint8) - eroded.astype(np.uint8)

    # Get coordinates of surface points
    surface_points = np.argwhere(surface > 0)

    return surface_points

def calculate_hd95(pred, target, percentile=95):
    """
    Calculate 95th percentile Hausdorff Distance (HD95)

    HD95 is calculated as:
    1. Get surface points of both prediction and ground truth
    2. For each surface point in pred, find the minimum distance to target surface
    3. For each surface point in target, find the minimum distance to pred surface
    4. Take the 95th percentile of all these distances

    Args:
        pred: Binary prediction mask (numpy array)
        target: Binary ground truth mask (numpy array)
        percentile: Percentile for Hausdorff distance (default: 95)

    Returns:
        HD95 value (float)
    """
    # Convert to binary if not already
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)

    # Check if both pred and target are empty
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0

    # If one is empty and the other is not, return a large distance
    if pred.sum() == 0 or target.sum() == 0:
        # Return a large but finite value (e.g., image diagonal)
        max_distance = np.sqrt(sum(dim**2 for dim in pred.shape))
        return float(max_distance)

    try:
        # Get surface points
        pred_surface = get_surface_points(pred)
        target_surface = get_surface_points(target)

        if len(pred_surface) == 0 or len(target_surface) == 0:
            # If no surface points, return 0
            return 0.0

        # Use KDTree for efficient distance calculation
        # Calculate distances from pred surface to target surface
        target_tree = cKDTree(target_surface)
        distances_pred_to_target, _ = target_tree.query(pred_surface, k=1)

        # Calculate distances from target surface to pred surface
        pred_tree = cKDTree(pred_surface)
        distances_target_to_pred, _ = pred_tree.query(target_surface, k=1)

        # Combine all distances (cKDTree.query returns numpy array)
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])

        # Calculate 95th percentile
        hd95 = np.percentile(all_distances, percentile)

        # Handle NaN or Inf values
        if np.isnan(hd95) or np.isinf(hd95):
            return 0.0

        return float(hd95)

    except Exception as e:
        print(f"Warning: Error calculating HD95: {e}")
        # Return a default value if calculation fails
        max_distance = np.sqrt(sum(dim**2 for dim in pred.shape))
        return float(max_distance)

def calculate_dice_intervals(dice_scores):
    """Calculate distribution of Dice scores across intervals"""
    intervals = [
        (0.0, 0.1),   # 0~0.1
        (0.1, 0.2),   # 0.1~0.2
        (0.2, 0.3),   # 0.2~0.3
        (0.3, 0.4),   # 0.3~0.4
        (0.4, 0.5),   # 0.4~0.5
        (0.5, 1.0)    # 0.5~1.0
    ]
    
    interval_counts = {}
    for i, (lower, upper) in enumerate(intervals):
        if i == len(intervals) - 1:  # Last interval includes upper bound
            count = sum(lower <= score <= upper for score in dice_scores)
        else:
            count = sum(lower <= score < upper for score in dice_scores)
        interval_name = f"{lower:.1f}-{upper:.1f}"
        interval_counts[interval_name] = count
    
    return interval_counts

def keep_largest_connected_component_3d(pred_data):
    """Keep only the largest connected component in 3D prediction volume"""
    import SimpleITK as sitk
    mask_img = sitk.GetImageFromArray(pred_data)
    cc = sitk.ConnectedComponent(mask_img, True)  
    cc_sorted = sitk.RelabelComponent(cc, sortByObjectSize=True)  # 按体积排序，最大区域编号=1
    largest_cc = sitk.Equal(cc_sorted, 1)                         # 保留最大连通域
    return sitk.GetArrayFromImage(largest_cc).astype(np.uint8)

def fix_cdsa_net_checkpoint(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'gre_dcgf_' in key:
            # Replace 'gre_dcgf_X' with 'BiCR_X'
            new_key = key.replace('gre_dcgf_', 'BiCR_')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def build_model_from_path(model_path: str, device: torch.device):
    experiment_name = Path(model_path).parent.name
    if experiment_name in ("SegGland_BMA_FullPICAI",):
        return BMANet(
            dim_in=1,
            num_classes=2,
            depths=[2, 2, 8, 3],
            stem_dim=24,
            embed_dims=[24, 48, 96, 192],
            drop=0.1,
        ).to(device)
    if experiment_name in ("SegGland_AWUNet_FullPICAI",):
        return build_aw_unet(num_classes=2, num_res_units=2).to(device)
    if experiment_name in ("SegGland_UNETR_FullPICAI",):
        return UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(16, 256, 256),
            spatial_dims=3,
        ).to(device)
    if experiment_name in ("SegGland_UNet_FullPICAI",):
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
            channels=[32, 64, 128, 256, 512, 1024],
        ).to(device)
    if experiment_name in ("SegGland_nnUNet_FullPICAI",):
        model, _, _ = build_generic_unet_from_plans(
            plans_file="/home/Space/clib/Projects/CAD/models/extracted_generic_unet/nnUNetPlansv2.1_plans_3D.pkl",
            deep_supervision=False,
        )
        return model.to(device)
    if experiment_name in ("SegGland_TransUNet_FullPICAI",):
        return build_transunet3d_main_output("R50-ViT-B_16").to(device)
    if experiment_name in ("SegGland_WaveFormer_FullPICAI",):
        return Waveformer(
            img_size=(16, 256, 256),
            patch_size=2,
            in_chans=1,
            out_chans=2,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            num_heads=[3, 6, 12, 24],
            drop_path_rate=0.1,
            use_checkpoint=False,
        ).to(device)
    if experiment_name in ("SegGland_WaTER_FullPICAI",):
        return WaTER_UNet3D(out_channels=2).to(device)
    if experiment_name in ("SegGland_DIY_PICAI_MSAGOnly", "SegGland_DIY_AHCDU_MSAGOnly"):
        return MSAGOnlyNet(in_channels=1, out_channels=2).to(device)
    if experiment_name in ("SegGland_DIY_PICAI_BoundaryOnly", "SegGland_DIY_AHCDU_BoundaryOnly"):
        return BoundaryOnlyNet(in_channels=1, out_channels=2).to(device)
    if experiment_name in ("SegGland_DIY_PICAI", "SegGland_DIY_AHCDU"):
        return DIYNoWave(in_channels=1, out_channels=2).to(device)
    if experiment_name in ("SegGland_DIY_PICAI_Wave", "SegGland_DIY_AHCDU_Wave"):
        return DIY(in_channels=1, out_channels=2).to(device)
    if experiment_name in ("SegGland_DIY_PICAI_Wave_Attentaion", "SegGland_DIY_AHCDU_Wave_Attentaion"):
        return DIYGate(in_channels=1, out_channels=2).to(device)
    if experiment_name in ("SegGland_DIY_PICAI_Wave_Attention_Bound", "SegGland_DIY_AHCDU_Wave_Attention_Bound"):
        return DIYBoundary(in_channels=1, out_channels=2).to(device)
    if experiment_name in (
        "SegGland_DIY_AHCDU_Wave_MSAGAttention_Bound_0.3",
        "SegGland_DIY_PICAI_Bound_Wave_MSAGAttention_0.2",
    ):
        return DIYBoundaryMSAG(in_channels=1, out_channels=2).to(device)

    return DIY(in_channels=1, out_channels=2).to(device)


def forward_model(model: torch.nn.Module, inputs: torch.Tensor):
    if isinstance(model, (DIYBoundary, DIYBoundaryMSAG, BoundaryOnlyNet)):
        return model(inputs, training=False)
    return model(inputs)


def main():
    
    
    # Set device
    if args.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    # model = UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=2,
    #     strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    #     channels=[32, 64, 128, 256, 512, 1024],
    #     ).to(device)
    
    # model = UNETR(in_channels=1, 
    #               out_channels=2, 
    #               img_size=(16, 256, 256), 
    #               spatial_dims=3).to(device)
    
    # model = AttentionUnet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=2,  # 背景和腺体两个类别
    #     channels=[16, 32, 64, 128, 256],  # 编码器通道数，从浅到深
    #     strides=[2, 2, 2, 2],  # 下采样步长，对应4次下采样
    #     kernel_size=3,
    #     up_kernel_size=3,
    #     dropout=0.1
    # ).to(device)
    
    model = build_model_from_path(args.model_path, device)
    # model, _, _ = build_generic_unet_from_plans(
    # plans_file="/home/Space/clib/Projects/CAD/models/extracted_generic_unet/nnUNetPlansv2.1_plans_3D.pkl",
    # deep_supervision=False)
    # model = model.to(device)
    
    # model = BMANet(
    #     dim_in=1, 
    #     num_classes=2,
    #     depths=[2, 2, 8, 3], 
    #     stem_dim=24, 
    #     embed_dims=[24, 48, 96, 192], 
    #     drop=0.1).to(device)
    
    # model = Waveformer(
    #     img_size=(16, 256, 256),
    #     patch_size=2,
    #     in_chans=1,
    #     out_chans=2,
    #     depths=[2, 2, 2, 2],
    #     feat_size=[48, 96, 192, 384],
    #     num_heads=[3, 6, 12, 24],
    #     drop_path_rate=0.1,
    #     use_checkpoint=False,
    # ).to(device)
    
    # model = build_transunet3d_main_output("R50-ViT-B_16").to(device)
    
    # model = VNet(ResGNet).to(device)
    
    # model = build_aw_unet(num_classes=2, num_res_units=2).to(device)
    
    # model = WaTER_UNet3D(out_channels=2).to(device)
    
    
    
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    
    checkpoint = checkpoint["model_state_dict"]
    
    checkpoint = fix_cdsa_net_checkpoint(checkpoint)  # 注释即可
    
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from: {args.model_path}")

    model_complexity = calculate_model_complexity(model, device)
    print(
        f"Model complexity @ {model_complexity['input_shape']}: "
        f"Params {model_complexity['params_m']:.2f}M, "
        f"FLOPs {model_complexity['flops_g']:.2f}G"
    )
    
    # Create parent directory based on model_path experiment name and mode
    model_path_parts = Path(args.model_path).parts
    if len(model_path_parts) >= 2:
        experiment_name = model_path_parts[-2]  # Get the experiment name from model path
    else:
        experiment_name = Path(args.model_path).stem  # Fallback to model file name
    

    suffix = "_infer" if args.mode == "val" else "_test"
    infer_root = "infer_FullPICAI" if "FullPICAI" in experiment_name or "FullPICAI" in str(args.data_path) else "infer"
    parent_dir = Path(f"{infer_root}/{experiment_name}{suffix}")
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Create output directories as subdirectories of parent directory
    output_dir = parent_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    slices_dir = parent_dir / args.slices_dir
    slices_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset based on mode # nnUNet_val
    if args.mode == "val":
        if args.data_path.endswith('PI-CAI'):
            images_dir = 'imagesTs'
            labels_dir = 'labelsTs'
        elif args.data_path.endswith('ChengdaOnlyCSPca'):
            images_dir = 'nnUNet_val/imagesTs'
            labels_dir = 'nnUNet_val/labelsTs'
        else:
            images_dir = 'imagesTs'
            labels_dir = 'labelsTs'
    else:  # test mode
        images_dir = 'imagesTs'
        labels_dir = 'labelsTs'
    
    val_dataset = Lits_DataSet(
        Path(args.data_path), 
        images_dir, 
        labels_dir,
        enable_augmentation=False
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=1,  # Batch size 1 for individual patient inference
        num_workers=args.num_workers, 
        shuffle=False
    )
    
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Test loop
    all_dice_scores = []
    all_miou_scores = []
    all_hd95_scores = []
    all_sensitivity_scores = []
    all_precision_scores = []
    case_metrics = []  # Store metrics for each individual case
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc='Testing')):
            inputs, labels, patient_names = batch_data
            inputs = inputs.to(device)         
            labels = labels.to(device)
            
            # Forward pass
            outputs = forward_model(model, inputs)
            
            # Apply softmax and get predictions
            outputs_softmax = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
            
            # preds = (outputs > 0.5).float()
            
            # Save prediction as NIfTI file
            patient_name = patient_names[0]
            pred_numpy = preds.squeeze().cpu().numpy().astype(np.uint8)
            
            # Conditionally keep only largest connected component
            if args.keep_largest_cc:
                pred_numpy_cleaned = keep_largest_connected_component_3d(pred_numpy)
                # Calculate metrics AFTER cleaning (using cleaned predictions)
                preds_cleaned = torch.from_numpy(pred_numpy_cleaned).unsqueeze(0).unsqueeze(0).to(device)
                dice_score = calculate_dice(preds_cleaned, labels)
                miou_score = calculate_miou(preds_cleaned, labels)
                sensitivity_score, precision_score = calculate_sensitivity_precision(preds_cleaned, labels)
                # Calculate HD95 using cleaned predictions
                gt_data = labels.squeeze().cpu().numpy().astype(np.uint8)
                hd95_score = calculate_hd95(pred_numpy_cleaned, gt_data, percentile=95)
                # Use cleaned predictions for saving
                pred_numpy_to_save = pred_numpy_cleaned
            else:
                # Calculate metrics using original predictions
                dice_score = calculate_dice(preds, labels)
                miou_score = calculate_miou(preds, labels)
                sensitivity_score, precision_score = calculate_sensitivity_precision(preds, labels)
                # Calculate HD95 using original predictions
                gt_data = labels.squeeze().cpu().numpy().astype(np.uint8)
                hd95_score = calculate_hd95(pred_numpy, gt_data, percentile=95)
                # Use original predictions for saving
                pred_numpy_to_save = pred_numpy
            
            all_dice_scores.append(dice_score)
            all_miou_scores.append(miou_score)
            all_hd95_scores.append(hd95_score)
            all_sensitivity_scores.append(sensitivity_score)
            all_precision_scores.append(precision_score)

            # Store individual case metrics (rounded to 4 decimal places)
            case_metrics.append({
                'patient_name': patient_name,
                'dice_score': round(dice_score, 4),
                'miou_score': round(miou_score, 4),
                'hd95_score': round(hd95_score, 4),
                'sensitivity_score': round(sensitivity_score, 4),
                'precision_score': round(precision_score, 4)
            })
            
            # Create NIfTI image (save predictions)
            # 获取原始图像的方向矩阵和形状
            try:
                # 构建原始图像路径（使用T2W图像获取方向矩阵）
                original_img_path = Path(args.data_path) / images_dir / f"{patient_name}_0000.nii.gz"
                if original_img_path.exists():
                    original_img = nib.load(original_img_path)
                    affine_matrix = original_img.affine
                    original_shape = original_img.shape
                    # print(f"原始图像形状: {original_shape}, 方向矩阵: {affine_matrix}")
                else:
                    print(f"警告: 原始图像不存在: {original_img_path}")
                    # 尝试其他可能的路径
                    alt_path = Path(args.data_path) / "imagesTr" / f"{patient_name}_0000.nii.gz"
                    if alt_path.exists():
                        original_img = nib.load(alt_path)
                        affine_matrix = original_img.affine
                        original_shape = original_img.shape
                        # print(f"使用备用路径方向矩阵: {affine_matrix}")
                    else:
                        print(f"警告: 备用图像也不存在: {alt_path}")
                        affine_matrix = np.eye(4)
                        original_shape = None
            except Exception as e:
                print(f"警告: 获取方向矩阵失败: {e}")
                affine_matrix = np.eye(4)
                original_shape = None

            # 检查预测数据的形状并转置为原始顺序
            # SimpleITK的sitk.GetArrayFromImage()返回[z, y, x] (D, H, W)
            # 但原始NIfTI文件的数据顺序是[x, y, z] (W, H, D)
            # 需要将预测从[D, H, W]转置为[W, H, D]
            if pred_numpy_to_save.ndim == 3:
                # 转置: [D, H, W] -> [W, H, D]
                pred_data_to_save = np.transpose(pred_numpy_to_save, (2, 1, 0))
                if original_shape is not None and pred_data_to_save.shape != original_shape:
                    print(f"警告: 转置后预测形状 {pred_data_to_save.shape} 与原始形状 {original_shape} 不匹配")
            else:
                print(f"警告: 预测数据维度不是3D: {pred_numpy_to_save.shape}")
                pred_data_to_save = pred_numpy_to_save

            pred_img = nib.Nifti1Image(pred_data_to_save, affine_matrix)
            pred_filename = output_dir / f"{patient_name}_pred.nii.gz"
            nib.save(pred_img, pred_filename)
            
            print(
                f"Patient: {patient_name}, Dice: {dice_score*100:.2f}, "
                f"mIoU: {miou_score*100:.2f}, HD95: {hd95_score:.4f}, "
                f"Sensitivity: {sensitivity_score*100:.2f}, "
                f"Precision: {precision_score*100:.2f}"
            )
    
    # Calculate overall metrics (rounded to 4 decimal places)
    avg_dice = round(np.mean(all_dice_scores), 4)
    avg_miou = round(np.mean(all_miou_scores), 4)
    avg_hd95 = round(np.mean(all_hd95_scores), 4)
    avg_sensitivity = round(np.mean(all_sensitivity_scores), 4)
    avg_precision = round(np.mean(all_precision_scores), 4)
    std_dice = round(np.std(all_dice_scores), 4)
    std_miou = round(np.std(all_miou_scores), 4)
    std_hd95 = round(np.std(all_hd95_scores), 4)
    std_sensitivity = round(np.std(all_sensitivity_scores), 4)
    std_precision = round(np.std(all_precision_scores), 4)

    print(f"\n=== Overall Results ===")
    print(f"Connected component filtering: {'Enabled' if args.keep_largest_cc else 'Disabled'}")
    print(f"Average Dice: {avg_dice*100:.2f} ± {std_dice*100:.2f}")
    print(f"Average mIoU: {avg_miou*100:.2f} ± {std_miou*100:.2f}")
    print(f"Average HD95: {avg_hd95:.4f} ± {std_hd95:.4f}")
    print(f"Average Sensitivity: {avg_sensitivity*100:.2f} ± {std_sensitivity*100:.2f}")
    print(f"Average Precision: {avg_precision*100:.2f} ± {std_precision*100:.2f}")
    print(f"All results saved to parent directory: {parent_dir}")
    print(f"- Predictions saved to: {output_dir}")
    print(f"- Slice images saved to: {slices_dir}")
    
    # Calculate Dice score intervals
    dice_interval_stats = calculate_dice_intervals([case['dice_score'] for case in case_metrics])
    
    # Save individual case metrics to separate A_Summary.json file
    summary_data = {
        'model_path': args.model_path,
        'data_path': args.data_path,
        'keep_largest_cc': args.keep_largest_cc,
        'model_complexity': model_complexity,
        'num_cases': len(case_metrics),
        'overall_metrics': {
            'dice': format_mean_std(avg_dice, std_dice, scale=100.0),
            'miou': format_mean_std(avg_miou, std_miou, scale=100.0),
            'hd95': format_mean_std(avg_hd95, std_hd95),
            'sensitivity': format_mean_std(avg_sensitivity, std_sensitivity, scale=100.0),
            'precision': format_mean_std(avg_precision, std_precision, scale=100.0)
        },
        'dice_interval_statistics': dice_interval_stats,
        'cases': case_metrics
    }
    
    summary_file = output_dir / '1A_Summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=4)
    
    print(f"Case summary saved to: {summary_file}")

    print("\n=== Phase 1 complete: metrics and predictions are ready ===")
    if args.skip_plots:
        print("Skipping contour image generation (--skip_plots).")
        return

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    renderer = Path(__file__).resolve().parent / 'render_prostate_contours.py'
    render_command = [
        sys.executable,
        str(renderer),
        '--data_path',
        str(args.data_path),
        '--images_dir',
        images_dir,
        '--labels_dir',
        labels_dir,
        '--predictions_dir',
        str(output_dir),
        '--output_dir',
        str(slices_dir),
        '--workers',
        str(args.plot_workers),
    ]
    print(f"\n=== Phase 2: generating contour images with {args.plot_workers} processes ===")
    render_result = subprocess.run(render_command, check=False)
    if render_result.returncode != 0:
        print(f"Warning: contour renderer exited with code {render_result.returncode}")

if __name__ == "__main__":
    main()
