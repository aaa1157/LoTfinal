"""
深度学习训练工具

支持 mode: debug / medium / full
使用 AdamW + CosineAnnealingLR + early stopping

依赖 torch
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Tuple

from src.config import dl_modes, dl_learning_rate, dl_weight_decay
from src.deep_models import get_model, count_parameters
from src.dataset import load_dataset, filter_by_strategy, filter_by_task, get_split_indices


class PhaseDataset(Dataset):
    """PyTorch Dataset for phase sequences"""

    def __init__(self, phases: np.ndarray, labels: np.ndarray):
        self.phases = torch.FloatTensor(phases)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.phases[idx].unsqueeze(0)  # [1, 1800]
        y = self.labels[idx]
        return x, y


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """评估"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def train_model(
    model_name: str,
    train_phases: np.ndarray,
    train_labels: np.ndarray,
    val_phases: np.ndarray,
    val_labels: np.ndarray,
    mode: str = "debug",
    device_name: str = 'auto',
    save_dir: str = 'results/models',
    log_dir: str = 'results/logs',
    seed: int = 2026,
) -> Dict:
    """
    训练深度学习模型

    Returns:
        dict with model, metrics, train_history
    """
    # Device
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    # Mode config
    mode_config = dl_modes.get(mode, dl_modes["debug"])
    epochs = mode_config["epochs"]
    patience = mode_config["patience"]
    batch_size = mode_config["batch_size"]

    # Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Model
    model = get_model(model_name).to(device)
    n_params = count_parameters(model)

    # Data
    train_ds = PhaseDataset(train_phases, train_labels)
    val_ds = PhaseDataset(val_phases, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=dl_learning_rate, weight_decay=dl_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience_counter = 0
    train_history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, f'{model_name}_best.pt'),
                                      map_location=device))

    # Inference time
    model.eval()
    dummy = torch.randn(1, 1, 1800).to(device)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    inference_time_ms = (time.time() - start) / 100 * 1000

    return {
        'model': model,
        'n_params': n_params,
        'best_val_acc': best_val_acc,
        'epochs_trained': len(train_history),
        'inference_time_ms': inference_time_ms,
        'device': str(device),
        'train_history': train_history,
    }


def test_model(
    model: nn.Module,
    test_phases: np.ndarray,
    test_labels: np.ndarray,
    device_name: str = 'auto',
) -> Dict:
    """测试模型"""
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    model = model.to(device)
    model.eval()

    test_ds = PhaseDataset(test_phases, test_labels)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    _, acc, preds, labels = evaluate(model, test_loader, criterion, device)

    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
    }
