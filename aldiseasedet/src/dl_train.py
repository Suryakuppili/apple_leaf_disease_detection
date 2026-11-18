import os
import time
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from src.dl_dataset import AppleLeafDataset
from src.dl_models.custom_cnn import get_model as get_custom_cnn
from src.dl_models.mobilenet_v2 import get_model as get_mobilenet_v2

# -------------------- Utility functions --------------------

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_augmentations():
    from torchvision import transforms
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1),
    ])

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets.detach().cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def save_checkpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)

def plot_curves(train_losses, val_losses, train_accs, val_accs, model_name, out_dir):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'training_curves_{model_name}.png')
    plt.savefig(out_path)
    plt.close()
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Train DL models for Apple leaf disease classification')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision if GPU available')
    parser.add_argument('--output-dir', type=str, default='models/dl')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # Prepare datasets & loaders
    train_dataset = AppleLeafDataset(split='train')
    train_dataset.set_train_augment(get_augmentations())
    val_dataset = AppleLeafDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_dataset.classes)
    models_info = {
        'custom_cnn': get_custom_cnn(num_classes),
        'mobilenet_v2': get_mobilenet_v2(num_classes),
    }

    # Training loop for each model
    for model_name, model in models_info.items():
        print(f'\n=== Training {model_name} ===')
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        start_time = time.time()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch:02d} | Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f} | {epoch_time:.1f}s')
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                checkpoint_path = os.path.join(args.output_dir, f'{model_name}.pt')
                save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                }, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f'Early stopping triggered after {epoch} epochs.')
                    break
        total_time = time.time() - start_time
        curve_path = plot_curves(train_losses, val_losses, train_accs, val_accs, model_name, args.results_dir)
        # Record training metadata
        model_file = os.path.join(args.output_dir, f'{model_name}.pt')
        model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
        metadata = {
            'model_name': model_name,
            'training_time_seconds': total_time,
            'model_size_mb': round(model_size_mb, 2),
            'training_curve_path': curve_path,
        }
        meta_path = os.path.join(args.results_dir, f'{model_name}_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f'Training of {model_name} finished in {total_time:.1f}s, model size {model_size_mb:.2f} MB')

if __name__ == '__main__':
    main()
