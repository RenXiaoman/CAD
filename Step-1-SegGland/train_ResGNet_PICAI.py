import os
import random
import re
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join
import json
import time


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
    
#### Model
from models.ResGNet import ResGNet, VNet
from Options.Options_ResGNet import Options_ResGNet_PICAI


from gland_dataset_nnunet import Lits_DataSet



def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    """
    Polynomial learning rate decay from nnUNet
    lr = initial_lr * (1 - epoch/max_epochs)^exponent
    """
    return initial_lr * (1 - epoch / max_epochs)**exponent


def resolve_resume_checkpoint(opt):
    """Resolve --resume to a checkpoint path, supporting True as task latest."""
    default_checkpoint = Path(opt.checkpoints_dir) / opt.task_name / 'model_latest.pth'

    if isinstance(opt.resume, bool):
        if not opt.resume:
            return None
        return default_checkpoint

    if opt.resume is None:
        return None

    resume = str(opt.resume).strip()
    if resume.lower() in ('', 'false', '0', 'none', 'no'):
        return None
    if resume.lower() in ('true', '1', 'yes'):
        return default_checkpoint

    return Path(resume)


def load_metric_history(metrics_file):
    history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_dices': [],
        'val_dices': [],
        'train_mious': [],
        'val_mious': [],
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
            history['epochs'].append(int(match.group('epoch')))
            history['train_losses'].append(float(match.group('train_loss')))
            history['val_losses'].append(float(match.group('val_loss')))
            history['train_dices'].append(float(match.group('train_dice')))
            history['val_dices'].append(float(match.group('val_dice')))
            history['train_mious'].append(float(match.group('train_miou')))
            history['val_mious'].append(float(match.group('val_miou')))

    return history


def main(options_class=Options_ResGNet_PICAI, train_augmentation=False):
    # Parse options
    opt_parser = options_class()
    opt = opt_parser.parse()
    
    # Set device
    if opt.gpu_ids != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Initialize model
    model = VNet(ResGNet).to(device)
    
    # Resume training from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_dice = 0.0
    best_val_miou = 0.0
    best_loss_epoch = 0
    best_dice_epoch = 0
    best_miou_epoch = 0
    
    # Define optimizer (nnUNet style: SGD with momentum 0.99)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    
    resume_path = resolve_resume_checkpoint(opt)
    if resume_path is not None:
        if resume_path.exists():
            print(f"Resuming training from: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            best_val_dice = checkpoint['best_val_dice']
            best_val_miou = checkpoint['best_val_miou']
            best_loss_epoch = checkpoint['best_loss_epoch']
            best_dice_epoch = checkpoint['best_dice_epoch']
            best_miou_epoch = checkpoint['best_miou_epoch']
            
            # Restore learning rate from checkpoint
            if 'current_lr' in checkpoint:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = checkpoint['current_lr']
                print(f"Resumed learning rate: {checkpoint['current_lr']:.6f}")
            
            print(f"Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Checkpoint {resume_path} not found, starting from scratch")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Metrics calculation functions
    def calculate_dice_loss(outputs, targets):
        """Official ResGNet-style Dice loss on single-channel probabilities."""
        smooth = 1e-6
        outputs_flat = outputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (outputs_flat * targets_flat).sum()
        return 1.0 - (2.0 * intersection + smooth) / (outputs_flat.sum() + targets_flat.sum() + smooth)

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
    
    def save_model_checkpoint(filename):
        """Save model checkpoint with all training metrics"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': avg_train_dice,
            'val_dice': avg_val_dice,
            'train_miou': avg_train_miou,
            'val_miou': avg_val_miou,
            'best_val_loss': best_val_loss,
            'best_val_dice': best_val_dice,
            'best_val_miou': best_val_miou,
            'best_loss_epoch': best_loss_epoch,
            'best_dice_epoch': best_dice_epoch,
            'best_miou_epoch': best_miou_epoch,
            'current_lr': new_lr
        }, results_dir / filename)
    
    def plot_result_mix(epochs, plot_data1, plot_data2, label1, label2, description, save_path, save_name, showCurrentBestLoss=None):
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, plot_data1, label=label1)
        plt.plot(epochs, plot_data2, label=label2)
        
        if showCurrentBestLoss is not None:
            plt.title(str(description)+f' ({showCurrentBestLoss})')
        else:
            plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(f'{save_name}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(join(save_path, f'{save_name}.png'))
        plt.close()
    

    # Create results directory
    results_dir = Path(opt.checkpoints_dir) / opt.task_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = results_dir / 'training_metrics.txt'
    metric_history = load_metric_history(metrics_file) if start_epoch > 0 else {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_dices': [],
        'val_dices': [],
        'train_mious': [],
        'val_mious': [],
    }
    epochs = metric_history['epochs']
    train_losses = metric_history['train_losses']
    val_losses = metric_history['val_losses']
    train_dices = metric_history['train_dices']
    val_dices = metric_history['val_dices']
    train_mious = metric_history['train_mious']
    val_mious = metric_history['val_mious']
    
    # Save training configuration
    config_file = results_dir / 'training_config.json'
    with open(config_file, 'w') as f:
        json.dump(vars(opt), f, indent=4)
    
    # Load dataset with augmentation for training, no augmentation for validation
    train_dataset = Lits_DataSet(Path(opt.datapath),
                                 'imagesTr', 
                                 'labelsTr', 
                                 enable_augmentation=train_augmentation)
    val_dataset = Lits_DataSet(Path(opt.datapath), 
                               'imagesTs', 
                               'labelsTs',
                               enable_augmentation=False)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_threads, shuffle=False)
    
    print(f'Train dataset: {len(train_dataset)} samples, Val dataset: {len(val_dataset)} samples')

    # Training loop
    for epoch in range(start_epoch, opt.epoch):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        all_train_dice_scores = []  # 存储每个样本的Dice分数
        all_train_miou_scores = []  # 存储每个样本的mIoU分数
        
        for batch_idx, batch_data in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Train]')):
            inputs, labels, patient_names = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Calculate loss
            loss = calculate_dice_loss(outputs, labels)
            
            # ResGNet outputs single-channel foreground probabilities.
            preds = (outputs > 0.5).float()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 计算每个样本的Dice和mIoU（case级别）
            batch_size = preds.size(0)
            for i in range(batch_size):
                single_pred = preds[i:i+1]
                single_label = labels[i:i+1]
                dice_score = calculate_dice(single_pred, single_label)
                miou_score = calculate_miou(single_pred, single_label)
                all_train_dice_scores.append(dice_score)
                all_train_miou_scores.append(miou_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        val_miou = 0
        all_val_dice_scores = []  # 存储每个样本的Dice分数
        all_val_miou_scores = []  # 存储每个样本的mIoU分数
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{opt.epoch} [Val]')):
                inputs, labels, patient_names = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                
                # Calculate loss
                loss = calculate_dice_loss(outputs, labels)
                
                # ResGNet outputs single-channel foreground probabilities.
                preds = (outputs > 0.5).float()

                val_loss += loss.item()
                
                # 计算每个样本的Dice和mIoU（case级别）
                batch_size = preds.size(0)
                for i in range(batch_size):
                    single_pred = preds[i:i+1]
                    single_label = labels[i:i+1]
                    dice_score = calculate_dice(single_pred, single_label)
                    miou_score = calculate_miou(single_pred, single_label)
                    all_val_dice_scores.append(dice_score)
                    all_val_miou_scores.append(miou_score)
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_dice = sum(all_train_dice_scores) / len(all_train_dice_scores)  # 样本级别平均
        avg_train_miou = sum(all_train_miou_scores) / len(all_train_miou_scores)  # 样本级别平均
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_dice = sum(all_val_dice_scores) / len(all_val_dice_scores)  # 真正的样本级别平均
        avg_val_miou = sum(all_val_miou_scores) / len(all_val_miou_scores)  # 真正的样本级别平均
        
        # Store metrics
        epochs.append(epoch + 1)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_dices.append(avg_train_dice)
        val_dices.append(avg_val_dice)
        train_mious.append(avg_train_miou)
        val_mious.append(avg_val_miou)
        
        # Update learning rate using poly_lr (nnUNet style)
        new_lr = poly_lr(epoch + 1, opt.epoch, opt.lr, 0.9)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch [{epoch+1}/{opt.epoch}], Time: {epoch_time:.2f}s, LR: {new_lr:.6f}, Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Train mIoU: {avg_train_miou:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val mIoU: {avg_val_miou:.4f}')
        
        # Save metrics to file
        with open(metrics_file, 'a') as f:
            f.write(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Train Dice: {avg_train_dice:.6f}, Train mIoU: {avg_train_miou:.6f}, Val Loss: {avg_val_loss:.6f}, Val Dice: {avg_val_dice:.6f}, Val mIoU: {avg_val_miou:.6f}\n')
        
        # Update best validation metrics and save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss_epoch = epoch + 1
            save_model_checkpoint('best_loss_model.pth')
        
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_dice_epoch = epoch + 1
            save_model_checkpoint('best_dice_model.pth')
        
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            best_miou_epoch = epoch + 1
        
        # Always save latest model for resume training
        save_model_checkpoint('model_latest.pth')
        
        # Save current model
        if (epoch + 1) % opt.model_save_fre == 0:
            save_model_checkpoint(f'model_epoch_{epoch+1}.pth')
        
        # Plot results every epoch
        plot_result_mix(epochs, train_losses, val_losses, 'Train_Loss', 'Val_Loss', 'Training and Validation Loss', str(results_dir), 'Loss_Curve', f'Best Loss: {best_val_loss:.6f} (Epoch {best_loss_epoch})')
        plot_result_mix(epochs, train_dices, val_dices, 'Train_Dice', 'Val_Dice', 'Training and Validation Dice', str(results_dir), 'Dice_Curve', f'Best Dice: {best_val_dice:.6f} (Epoch {best_dice_epoch})')
        plot_result_mix(epochs, train_mious, val_mious, 'Train_mIoU', 'Val_mIoU', 'Training and Validation mIoU', str(results_dir), 'mIoU_Curve', f'Best mIoU: {best_val_miou:.6f} (Epoch {best_miou_epoch})')
    
    print("Training completed!")


if __name__ == "__main__":
    main()
