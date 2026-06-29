"""
LMC 补充实验脚本

7.1 同 frozen PhaseCNN 下横向对比 (lmc_same_attacker_comparison)
7.2 LMC cross-model evaluation (lmc_cross_model_results)
7.3 LMC adaptive attacker (lmc_adaptive_attacker_results)

Usage:
    python run_lmc_experiments.py --mode medium --split random
    python run_lmc_experiments.py --mode medium --split scene_disjoint
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    RANDOM_SEED, fs as FS, strategies, lmc_modes,
)
from src.utils import make_rng, preprocess_phase, compute_phase_entropy, compute_spectral_entropy, compute_lowfreq_energy_ratio
from src.signal_model import make_time_axis, simulate_received_signal
from src.metasurface import generate_metasurface_signal, compute_switching_rate
from src.dataset import load_dataset, filter_by_strategy, filter_by_task, get_split_indices
from src.deep_models import get_model
from src.deep_train import PhaseDataset, train_model, test_model
from src.metrics import compute_phase_variance, compute_avg_phase_entropy
from src.learnable_controller import LMCController


def compute_balanced_accuracy(preds, labels):
    """计算 balanced accuracy"""
    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float((tpr + tnr) / 2)


def compute_auc(preds_proba, labels):
    """计算 AUC (使用正类概率)"""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, preds_proba))
    except Exception:
        return float('nan')


def evaluate_strategy_with_frozen_attacker(
    strategy_name, attacker_model, device,
    n_samples=500, seed=2026, params=None,
    display_name=None,
):
    """
    在同一冻结攻击者下评估某策略的防护效果
    返回详细指标

    Args:
        strategy_name: 实际策略名（传给 generate_metasurface_signal）
        display_name: CSV 中显示的策略名（如 lmc_controller）
        n_samples: 评估样本数（增加到 500 以提高统计可靠性）
    """
    t = make_time_axis()
    rng = make_rng(seed)

    all_preds = []
    all_labels = []
    all_proba = []
    phase_entropies = []
    spectral_entropies = []
    lf_ratios = []
    sw_rates = []

    for i in range(n_samples):
        sample_rng = make_rng(rng.integers(0, 2**31))
        motion_label = int(sample_rng.integers(0, 2))

        meta_rng = make_rng(sample_rng.integers(0, 2**31))
        v_meta, states, _ = generate_metasurface_signal(
            strategy_name, t, rng=meta_rng, params=params
        )

        _, phi, _ = simulate_received_signal(
            t, strategy_name=strategy_name,
            motion_label=motion_label,
            task_type="walking_detection",
            rng=sample_rng,
            meta_component=v_meta,
        )

        p = preprocess_phase(phi, detrend=True, zero_mean=True, standardize=True)
        x = torch.FloatTensor(p).unsqueeze(0).unsqueeze(0).to(device)

        attacker_model.eval()
        with torch.no_grad():
            out = attacker_model(x)
            prob = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1).item()
            prob_motion = prob[0, 1].item()

        all_preds.append(pred)
        all_labels.append(motion_label)
        all_proba.append(prob_motion)

        # 分别计算 phase_entropy 和 spectral_entropy
        phase_entropies.append(compute_phase_entropy(p))
        spectral_entropies.append(compute_spectral_entropy(p, FS))
        lf_ratios.append(compute_lowfreq_energy_ratio(p, FS))
        sw_rates.append(compute_switching_rate(states, FS))

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    proba = np.array(all_proba)

    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = compute_balanced_accuracy(preds, labels)
    flipped_acc = 1.0 - accuracy
    auc = compute_auc(proba, labels)

    notes = ""
    if accuracy < 0.5:
        notes = f"accuracy={accuracy:.3f}<0.5, flipped_accuracy={flipped_acc:.3f}, possible label flip"
    # 检测模型是否只预测一个类
    unique_preds = np.unique(preds)
    if len(unique_preds) == 1:
        notes += f" [WARNING: model always predicts class {unique_preds[0]}, degenerate classifier]"

    display = display_name if display_name else strategy_name

    return {
        'strategy': display,
        'frozen_attacker_model': 'PhaseCNN',
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'flipped_accuracy': float(flipped_acc),
        'TPR': float(tpr),
        'FPR': float(fpr),
        'TNR': float(tnr),
        'FNR': float(fnr),
        'precision': float(precision),
        'recall': float(recall),
        'F1': float(f1),
        'AUC': float(auc),
        'phase_entropy': float(np.mean(phase_entropies)),
        'spectral_entropy': float(np.mean(spectral_entropies)),
        'lowfreq_energy_ratio': float(np.mean(lf_ratios)),
        'switching_rate': float(np.mean(sw_rates)),
        'respiration_error': 0.0,
        'notes': notes.strip(),
    }


def run_same_attacker_comparison(mode, split_type, out_dir):
    """7.1 同 frozen PhaseCNN 下横向对比"""
    print("\n" + "=" * 60)
    print("  7.1 LMC Same-Attacker Comparison (frozen PhaseCNN)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load frozen PhaseCNN
    attacker = get_model('PhaseCNN').to(device)
    model_path = os.path.join(PROJECT_ROOT, 'results/models/PhaseCNN_best.pt')
    if os.path.exists(model_path):
        attacker.load_state_dict(torch.load(model_path, map_location=device))
    attacker.eval()

    # Load LMC best params
    lmc_params_path = os.path.join(PROJECT_ROOT, 'results/tables/lmc_best_params.csv')
    lmc_params = None
    if os.path.exists(lmc_params_path):
        df_params = pd.read_csv(lmc_params_path)
        lmc_params = {}
        for _, row in df_params.iterrows():
            val = row['value']
            try:
                lmc_params[row['param']] = eval(str(val))
            except Exception:
                lmc_params[row['param']] = val

    results = []
    for strat in strategies:
        print(f"  Evaluating: {strat}")
        r = evaluate_strategy_with_frozen_attacker(
            strat, attacker, device, n_samples=200, seed=RANDOM_SEED
        )
        results.append(r)
        print(f"    acc={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}, "
              f"AUC={r['AUC']:.3f}, F1={r['F1']:.3f}")

    # LMC controller
    if lmc_params is not None:
        print(f"  Evaluating: lmc_controller")
        r = evaluate_strategy_with_frozen_attacker(
            'multifreq_proposed', attacker, device, n_samples=500,
            seed=RANDOM_SEED, params=lmc_params,
            display_name='lmc_controller',
        )
        results.append(r)
        print(f"    acc={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}, "
              f"AUC={r['AUC']:.3f}, F1={r['F1']:.3f}")

    df = pd.DataFrame(results)
    path = os.path.join(out_dir, 'lmc_same_attacker_comparison.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def run_cross_model_evaluation(mode, split_type, out_dir):
    """7.2 LMC cross-model evaluation"""
    print("\n" + "=" * 60)
    print("  7.2 LMC Cross-Model Evaluation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load LMC best params
    lmc_params_path = os.path.join(PROJECT_ROOT, 'results/tables/lmc_best_params.csv')
    lmc_params = None
    if os.path.exists(lmc_params_path):
        df_params = pd.read_csv(lmc_params_path)
        lmc_params = {}
        for _, row in df_params.iterrows():
            val = row['value']
            try:
                lmc_params[row['param']] = eval(str(val))
            except Exception:
                lmc_params[row['param']] = val

    if lmc_params is None:
        print("  [SKIP] No LMC params found")
        return pd.DataFrame()

    model_names = ['PhaseCNN', 'PhaseNetLite', 'TinyTCN', 'ResNet1DLite', 'DualBranchNet']
    results = []

    for model_name in model_names:
        print(f"  Testing with frozen attacker: {model_name}")
        attacker = get_model(model_name).to(device)
        model_path = os.path.join(PROJECT_ROOT, 'results/models', f'{model_name}_best.pt')
        if os.path.exists(model_path):
            attacker.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"    [WARNING] No saved model for {model_name}, using random init")
        attacker.eval()

        r = evaluate_strategy_with_frozen_attacker(
            'multifreq_proposed', attacker, device, n_samples=500,
            seed=RANDOM_SEED, params=lmc_params,
            display_name='lmc_controller',
        )
        r['attacker_model'] = model_name
        results.append(r)
        print(f"    acc={r['accuracy']:.3f}, balanced_acc={r['balanced_accuracy']:.3f}, "
              f"AUC={r['AUC']:.3f}")

    df = pd.DataFrame(results)
    path = os.path.join(out_dir, 'lmc_cross_model_results.csv')
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return df


def run_adaptive_attacker(mode, split_type, seeds, out_dir):
    """7.3 LMC adaptive attacker"""
    print("\n" + "=" * 60)
    print("  7.3 LMC Adaptive Attacker")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load LMC best params
    lmc_params_path = os.path.join(PROJECT_ROOT, 'results/tables/lmc_best_params.csv')
    lmc_params = None
    if os.path.exists(lmc_params_path):
        df_params = pd.read_csv(lmc_params_path)
        lmc_params = {}
        for _, row in df_params.iterrows():
            val = row['value']
            try:
                lmc_params[row['param']] = eval(str(val))
            except Exception:
                lmc_params[row['param']] = val

    if lmc_params is None:
        print("  [SKIP] No LMC params found")
        return pd.DataFrame()

    # Build LMC dataset
    print("  Building LMC controller dataset...")
    from src.dataset import build_dataset

    # We need to add lmc_controller as a strategy in the dataset
    # Instead, we generate LMC data on-the-fly and train attackers
    t = make_time_axis()
    rng = make_rng(RANDOM_SEED)

    mode_config = {
        "debug": {"n_samples": 100},
        "medium": {"n_samples": 500},
        "full": {"n_samples": 1000},
    }
    n_per_label = mode_config.get(mode, mode_config["medium"])["n_samples"]

    all_phases = []
    all_labels = []

    for motion_label in [0, 1]:
        for i in range(n_per_label):
            sample_rng = make_rng(rng.integers(0, 2**31))
            meta_rng = make_rng(sample_rng.integers(0, 2**31))
            v_meta, _, _ = generate_metasurface_signal(
                'multifreq_proposed', t, rng=meta_rng, params=lmc_params
            )
            _, phi, _ = simulate_received_signal(
                t, strategy_name='lmc_controller',
                motion_label=motion_label,
                task_type="walking_detection",
                rng=sample_rng,
                meta_component=v_meta,
            )
            p = preprocess_phase(phi, detrend=True, zero_mean=True, standardize=True)
            all_phases.append(p)
            all_labels.append(motion_label)

    phases = np.array(all_phases)
    labels = np.array(all_labels)

    # Split
    n_total = len(labels)
    indices = np.arange(n_total)
    split_rng = make_rng(RANDOM_SEED + 1)
    split_rng.shuffle(indices)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)

    train_phases = phases[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    val_phases = phases[indices[n_train:n_train + n_val]]
    val_labels = labels[indices[n_train:n_train + n_val]]
    test_phases = phases[indices[n_train + n_val:]]
    test_labels = labels[indices[n_train + n_val:]]

    print(f"  LMC dataset: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")

    # Train attackers
    model_names = ['PhaseCNN', 'ResNet1DLite', 'DualBranchNet']
    all_results = []

    for seed in seeds:
        for model_name in model_names:
            print(f"  Training {model_name} on LMC data, seed={seed}...")
            result = train_model(
                model_name, train_phases, train_labels,
                val_phases, val_labels,
                mode=mode,
                save_dir=os.path.join(PROJECT_ROOT, 'results/final_results/checkpoints'),
                seed=seed,
            )

            test_metrics = test_model(result['model'], test_phases, test_labels)

            # Compute balanced accuracy and AUC
            test_ds = PhaseDataset(test_phases, test_labels)
            test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
            result['model'].eval()
            all_preds = []
            all_lbls = []
            all_proba = []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = result['model'](x)
                    prob = torch.softmax(out, dim=1)
                    all_preds.extend(out.argmax(dim=1).cpu().numpy())
                    all_lbls.extend(y.numpy())
                    all_proba.extend(prob[:, 1].cpu().numpy())

            preds = np.array(all_preds)
            lbls = np.array(all_lbls)
            proba = np.array(all_proba)
            balanced_acc = compute_balanced_accuracy(preds, lbls)
            flipped_acc = 1.0 - test_metrics['accuracy']
            auc = compute_auc(proba, lbls)

            row = {
                'model': model_name,
                'split_type': split_type,
                'seed': seed,
                'accuracy': test_metrics['accuracy'],
                'balanced_accuracy': balanced_acc,
                'flipped_accuracy': flipped_acc,
                'TPR': test_metrics['tpr'],
                'FPR': test_metrics['fpr'],
                'precision': test_metrics['precision'],
                'recall': test_metrics['recall'],
                'F1': test_metrics['f1'],
                'AUC': auc,
                'params': result['n_params'],
                'inference_time_ms': result['inference_time_ms'],
            }
            all_results.append(row)
            print(f"    acc={test_metrics['accuracy']:.3f}, balanced_acc={balanced_acc:.3f}, "
                  f"AUC={auc:.3f}, F1={test_metrics['f1']:.3f}")

    # Save raw results
    df_raw = pd.DataFrame(all_results)
    path_raw = os.path.join(out_dir, 'lmc_adaptive_attacker_results.csv')
    df_raw.to_csv(path_raw, index=False)
    print(f"  Saved: {path_raw}")

    # Save summary
    summary_rows = []
    for model_name in model_names:
        subset = df_raw[df_raw['model'] == model_name]
        if len(subset) > 0:
            summary_rows.append({
                'model': model_name,
                'split_type': split_type,
                'mean_accuracy': subset['accuracy'].mean(),
                'std_accuracy': subset['accuracy'].std() if len(subset) > 1 else 0.0,
                'mean_F1': subset['F1'].mean(),
                'std_F1': subset['F1'].std() if len(subset) > 1 else 0.0,
                'mean_AUC': subset['AUC'].mean(),
                'std_AUC': subset['AUC'].std() if len(subset) > 1 else 0.0,
                'num_seeds': len(subset),
            })

    df_sum = pd.DataFrame(summary_rows)
    path_sum = os.path.join(out_dir, 'lmc_adaptive_attacker_summary.csv')
    df_sum.to_csv(path_sum, index=False)
    print(f"  Saved: {path_sum}")

    return df_raw


def main():
    parser = argparse.ArgumentParser(description="LMC Supplementary Experiments")
    parser.add_argument('--mode', type=str, default='medium', choices=['debug', 'medium', 'full'])
    parser.add_argument('--split', type=str, default='random', choices=['random', 'scene_disjoint'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'same_attacker', 'cross_model', 'adaptive'])
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, 'results/final_results/tables')
    os.makedirs(out_dir, exist_ok=True)

    if args.experiment in ['all', 'same_attacker']:
        run_same_attacker_comparison(args.mode, args.split, out_dir)

    if args.experiment in ['all', 'cross_model']:
        run_cross_model_evaluation(args.mode, args.split, out_dir)

    if args.experiment in ['all', 'adaptive']:
        run_adaptive_attacker(args.mode, args.split, args.seeds, out_dir)

    print("\n  LMC experiments complete!")


if __name__ == '__main__':
    main()
