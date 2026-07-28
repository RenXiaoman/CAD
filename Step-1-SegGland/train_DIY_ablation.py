import argparse
import json
import os
import random
import re
import sys
import time
from os.path import join
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from monai.losses import DiceFocalLoss, DiceLoss  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gland_dataset_boundary import Boundary_DataSet
from gland_dataset_nnunet import Lits_DataSet
from models.DIY.ablation import BoundaryOnlyNet, MSAGOnlyNet


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


DATASET_DEFAULTS = {
    "AHCDU": {
        "datapath": "dataset/AHCDU/二分类图像/nnUNet_raw/Dataset130_ProstateAHCDU",
        "num_threads": 20,
        "gpu_ids": "0",
    },
    "PICAI": {
        "datapath": "dataset/PI-CAI/nnUNet_raw/Dataset131_ProstatePI-CAI",
        "num_threads": 20,
        "gpu_ids": "0",
    },
}


VARIANT_DEFAULTS = {
    "msag": {
        "task_suffix": "MSAGOnly",
        "batch_size": 10,
        "boundary": False,
    },
    "boundary": {
        "task_suffix": "BoundaryOnly",
        "batch_size": 6,
        "boundary": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DIY ablation variants for prostate gland segmentation."
    )
    parser.add_argument("--dataset", choices=("AHCDU", "PICAI"), required=True)
    parser.add_argument("--variant", choices=("msag", "boundary"), required=True)
    parser.add_argument("--datapath", type=str, default=None)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model_save_fre", type=int, default=50)
    parser.add_argument("--resume", default="True")
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--focal_weight", type=float, default=0.5)
    parser.add_argument("--boundary_weight", type=float, default=0.3)
    parser.add_argument("--boundary_radius", type=int, default=1)
    args = parser.parse_args()

    dataset_defaults = DATASET_DEFAULTS[args.dataset]
    variant_defaults = VARIANT_DEFAULTS[args.variant]
    args.datapath = args.datapath or dataset_defaults["datapath"]
    args.gpu_ids = args.gpu_ids or dataset_defaults["gpu_ids"]
    args.num_threads = args.num_threads or dataset_defaults["num_threads"]
    args.batch_size = args.batch_size or variant_defaults["batch_size"]
    if args.task_name is None:
        args.task_name = f"SegGland_DIY_{args.dataset}_{variant_defaults['task_suffix']}"
    return args


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def resolve_resume_checkpoint(opt):
    default_checkpoint = Path(opt.checkpoints_dir) / opt.task_name / "model_latest.pth"
    resume = str(opt.resume).strip()
    if resume.lower() in ("", "false", "0", "none", "no"):
        return None
    if resume.lower() in ("true", "1", "yes"):
        return default_checkpoint
    return Path(resume)


def load_metric_history(metrics_file):
    history = {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "train_dices": [],
        "val_dices": [],
        "train_mious": [],
        "val_mious": [],
    }
    if not metrics_file.exists():
        return history

    pattern = re.compile(
        r"Epoch (?P<epoch>\d+): Train Loss: (?P<train_loss>[-+0-9.eE]+), "
        r"Train Dice: (?P<train_dice>[-+0-9.eE]+), Train mIoU: (?P<train_miou>[-+0-9.eE]+), "
        r"Val Loss: (?P<val_loss>[-+0-9.eE]+), Val Dice: (?P<val_dice>[-+0-9.eE]+), "
        r"Val mIoU: (?P<val_miou>[-+0-9.eE]+)"
    )

    with open(metrics_file) as f:
        for line in f:
            match = pattern.search(line)
            if match is None:
                continue
            history["epochs"].append(int(match.group("epoch")))
            history["train_losses"].append(float(match.group("train_loss")))
            history["val_losses"].append(float(match.group("val_loss")))
            history["train_dices"].append(float(match.group("train_dice")))
            history["val_dices"].append(float(match.group("val_dice")))
            history["train_mious"].append(float(match.group("train_miou")))
            history["val_mious"].append(float(match.group("val_miou")))
    return history


def calculate_dice(preds, targets):
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    return ((2.0 * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)).item()


def calculate_miou(preds, targets):
    smooth = 1e-6
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    intersection = (preds_flat * targets_flat).sum()
    union = preds_flat.sum() + targets_flat.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def plot_result_mix(epochs, plot_data1, plot_data2, label1, label2, description, save_path, save_name, best_text):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, plot_data1, label=label1)
    plt.plot(epochs, plot_data2, label=label2)
    plt.title(f"{description} ({best_text})")
    plt.xlabel("Epoch")
    plt.ylabel(save_name)
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(join(save_path, f"{save_name}.png"))
    plt.close()


def build_model(variant, device):
    if variant == "msag":
        return MSAGOnlyNet(in_channels=1, out_channels=2).to(device)
    if variant == "boundary":
        return BoundaryOnlyNet(in_channels=1, out_channels=2).to(device)
    raise ValueError(f"Unknown ablation variant: {variant}")


def build_datasets(opt):
    if VARIANT_DEFAULTS[opt.variant]["boundary"]:
        train_dataset = Boundary_DataSet(
            Path(opt.datapath),
            "imagesTr",
            "labelsTr",
            enable_augmentation=True,
            boundary_radius=opt.boundary_radius,
        )
        val_dataset = Boundary_DataSet(
            Path(opt.datapath),
            "imagesTs",
            "labelsTs",
            enable_augmentation=False,
            boundary_radius=opt.boundary_radius,
        )
    else:
        train_dataset = Lits_DataSet(
            Path(opt.datapath),
            "imagesTr",
            "labelsTr",
            enable_augmentation=True,
        )
        val_dataset = Lits_DataSet(
            Path(opt.datapath),
            "imagesTs",
            "labelsTs",
            enable_augmentation=False,
        )
    return train_dataset, val_dataset


def main():
    opt = parse_args()

    if opt.gpu_ids != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = build_model(opt.variant, device)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)

    start_epoch = 0
    best_val_loss = float("inf")
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0

    resume_path = resolve_resume_checkpoint(opt)
    if resume_path is not None:
        if resume_path.exists():
            print(f"Resuming training from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            best_val_dice = checkpoint["best_val_dice"]
            best_val_miou = checkpoint["best_val_miou"]
            best_loss_epoch = checkpoint["best_loss_epoch"]
            best_dice_epoch = checkpoint["best_dice_epoch"]
            best_miou_epoch = checkpoint["best_miou_epoch"]
            if "current_lr" in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = checkpoint["current_lr"]
                print(f"Resumed learning rate: {checkpoint['current_lr']:.6f}")
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint {resume_path} not found, starting from scratch")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    dice_focal_loss = DiceFocalLoss(
        include_background=False,
        softmax=True,
        to_onehot_y=True,
        gamma=2.0,
        weight=[0.2, 0.8],
    ).to(device)
    boundary_dice_loss = DiceLoss(sigmoid=True).to(device)
    boundary_bce_loss = torch.nn.BCEWithLogitsLoss().to(device)

    results_dir = Path(opt.checkpoints_dir) / opt.task_name
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "training_metrics.txt"
    metric_history = load_metric_history(metrics_file) if start_epoch > 0 else {
        "epochs": [],
        "train_losses": [],
        "val_losses": [],
        "train_dices": [],
        "val_dices": [],
        "train_mious": [],
        "val_mious": [],
    }
    epochs = metric_history["epochs"]
    train_losses = metric_history["train_losses"]
    val_losses = metric_history["val_losses"]
    train_dices = metric_history["train_dices"]
    val_dices = metric_history["val_dices"]
    train_mious = metric_history["train_mious"]
    val_mious = metric_history["val_mious"]

    with open(results_dir / "training_config.json", "w") as f:
        json.dump(vars(opt), f, indent=4, ensure_ascii=False)

    train_dataset, val_dataset = build_datasets(opt)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_threads,
        shuffle=False,
    )
    print(f"Train dataset: {len(train_dataset)} samples, Val dataset: {len(val_dataset)} samples")

    def save_model_checkpoint(filename, epoch, metrics, current_lr):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": metrics["avg_train_loss"],
                "val_loss": metrics["avg_val_loss"],
                "train_dice": metrics["avg_train_dice"],
                "val_dice": metrics["avg_val_dice"],
                "train_miou": metrics["avg_train_miou"],
                "val_miou": metrics["avg_val_miou"],
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
                "best_val_miou": best_val_miou,
                "best_loss_epoch": best_loss_epoch,
                "best_dice_epoch": best_dice_epoch,
                "best_miou_epoch": best_miou_epoch,
                "current_lr": current_lr,
            },
            results_dir / filename,
        )

    use_boundary = VARIANT_DEFAULTS[opt.variant]["boundary"]
    for epoch in range(start_epoch, opt.epoch):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        all_train_dice_scores = []
        all_train_miou_scores = []

        for batch_data in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{opt.epoch} [Train]"):
            if use_boundary:
                inputs, labels, boundaries, patient_names = batch_data
                boundaries = boundaries.to(device).float()
            else:
                inputs, labels, patient_names = batch_data
                boundaries = None
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if use_boundary:
                outputs, boundary_outputs = model(inputs, training=True)
                seg_loss = dice_focal_loss(outputs, labels)
                boundary_loss = boundary_dice_loss(boundary_outputs, boundaries) + boundary_bce_loss(
                    boundary_outputs, boundaries
                )
                loss = seg_loss + opt.boundary_weight * boundary_loss
            else:
                outputs = model(inputs)
                loss = dice_focal_loss(outputs, labels)

            outputs_softmax = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            for i in range(preds.size(0)):
                all_train_dice_scores.append(calculate_dice(preds[i : i + 1], labels[i : i + 1]))
                all_train_miou_scores.append(calculate_miou(preds[i : i + 1], labels[i : i + 1]))

        model.eval()
        val_loss = 0
        all_val_dice_scores = []
        all_val_miou_scores = []

        with torch.no_grad():
            for batch_data in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{opt.epoch} [Val]"):
                if use_boundary:
                    inputs, labels, boundaries, patient_names = batch_data
                else:
                    inputs, labels, patient_names = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs, training=False) if use_boundary else model(inputs)
                loss = dice_focal_loss(outputs, labels)
                outputs_softmax = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs_softmax, dim=1, keepdim=True)
                val_loss += loss.item()

                for i in range(preds.size(0)):
                    all_val_dice_scores.append(calculate_dice(preds[i : i + 1], labels[i : i + 1]))
                    all_val_miou_scores.append(calculate_miou(preds[i : i + 1], labels[i : i + 1]))

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        avg_train_dice = sum(all_train_dice_scores) / len(all_train_dice_scores)
        avg_train_miou = sum(all_train_miou_scores) / len(all_train_miou_scores)
        avg_val_dice = sum(all_val_dice_scores) / len(all_val_dice_scores)
        avg_val_miou = sum(all_val_miou_scores) / len(all_val_miou_scores)

        epochs.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        train_mious.append(avg_train_miou)
        val_mious.append(avg_val_miou)

        new_lr = poly_lr(epoch + 1, opt.epoch, opt.lr, 0.9)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

        metrics = {
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_train_dice": avg_train_dice,
            "avg_val_dice": avg_val_dice,
            "avg_train_miou": avg_train_miou,
            "avg_val_miou": avg_val_miou,
        }
        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch + 1}/{opt.epoch}], Time: {epoch_time:.2f}s, LR: {new_lr:.6f}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, "
            f"Train mIoU: {avg_train_miou:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Val Dice: {avg_val_dice:.4f}, Val mIoU: {avg_val_miou:.4f}"
        )

        with open(metrics_file, "a") as f:
            f.write(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, "
                f"Train Dice: {avg_train_dice:.6f}, Train mIoU: {avg_train_miou:.6f}, "
                f"Val Loss: {avg_val_loss:.6f}, Val Dice: {avg_val_dice:.6f}, "
                f"Val mIoU: {avg_val_miou:.6f}\n"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_epoch = epoch + 1
            save_model_checkpoint("best_loss_model.pth", epoch, metrics, new_lr)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_dice_epoch = epoch + 1
            save_model_checkpoint("best_dice_model.pth", epoch, metrics, new_lr)
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_miou_epoch = epoch + 1

        save_model_checkpoint("model_latest.pth", epoch, metrics, new_lr)
        if (epoch + 1) % opt.model_save_fre == 0:
            save_model_checkpoint(f"model_epoch_{epoch + 1}.pth", epoch, metrics, new_lr)

        plot_result_mix(
            epochs,
            train_losses,
            val_losses,
            "Train_Loss",
            "Val_Loss",
            "Training and Validation Loss",
            str(results_dir),
            "Loss_Curve",
            f"Best Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch})",
        )
        plot_result_mix(
            epochs,
            train_dices,
            val_dices,
            "Train_Dice",
            "Val_Dice",
            "Training and Validation Dice",
            str(results_dir),
            "Dice_Curve",
            f"Best Dice: {best_val_dice:.6f} (Epoch {best_dice_epoch})",
        )
        plot_result_mix(
            epochs,
            train_mious,
            val_mious,
            "Train_mIoU",
            "Val_mIoU",
            "Training and Validation mIoU",
            str(results_dir),
            "mIoU_Curve",
            f"Best mIoU: {best_val_miou:.6f} (Epoch {best_miou_epoch})",
        )

    print("Training completed!")


if __name__ == "__main__":
    main()
