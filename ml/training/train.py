"""
Training script for the ASL classifier.

Usage:
    python -m training.train

This script:
  1. Loads the dataset
  2. Builds the CNN-Transformer model
  3. Trains with early stopping and LR scheduling
  4. Saves the best checkpoint and label map
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path so we can import from ml/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from models.classifier import ASLClassifier
from training.dataset import ASLImageDataset


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run evaluation. Returns average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def main():
    cfg = Config()

    device = torch.device(cfg.train.device)
    print(f"Using device: {device}")

    # --- Data ---
    train_dataset = ASLImageDataset(
        data_dir=cfg.data.processed_data_dir,
        split="train",
        image_size=cfg.data.image_size,
        augment=True,
    )
    val_dataset = ASLImageDataset(
        data_dir=cfg.data.processed_data_dir,
        split="val",
        image_size=cfg.data.image_size,
        augment=False,
    )

    if len(train_dataset) == 0:
        print("ERROR: No training data found. Run the data pipeline first.")
        print(f"  Expected data at: {cfg.data.processed_data_dir}/images/train/")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model = ASLClassifier(
        num_classes=cfg.model.num_classes,
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        d_model=cfg.model.transformer_dim,
        nhead=cfg.model.transformer_heads,
        num_encoder_layers=cfg.model.transformer_layers,
        dropout=cfg.model.transformer_dropout,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimizer & Scheduler ---
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

    # --- Training Loop ---
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_dir = Path(cfg.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        start_time = time.time()

        # Freeze/unfreeze backbone
        if epoch <= cfg.model.backbone_freeze_epochs:
            model.freeze_backbone()
        else:
            model.unfreeze_backbone()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d}/{cfg.train.epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"  -> New best model saved (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1

        # Periodic checkpoints
        if epoch % cfg.train.save_every_n_epochs == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"epoch_{epoch}.pt")

        # Early stopping
        if patience_counter >= cfg.train.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience={cfg.train.early_stopping_patience})")
            break

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
