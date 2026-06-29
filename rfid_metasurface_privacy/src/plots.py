"""
可视化模块

所有图保存到 results/figures/
使用 matplotlib，不使用 seaborn
不依赖 torch
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
import os

from src.utils import preprocess_phase, compute_spectrum


def _save_fig(fig, save_path: str, dpi: int = 200):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_phase_no_metasurface_motion_vs_nomotion(
    motion_phase: np.ndarray,
    nomotion_phase: np.ndarray,
    fs: float = 30.0,
    save_path: str = "results/figures/phase_no_metasurface_motion_vs_nomotion.png",
):
    """图1: 无超表面时 motion vs no_motion 相位对比"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = np.arange(len(motion_phase)) / fs

    axes[0].plot(t, preprocess_phase(motion_phase), linewidth=0.5, color='red', alpha=0.8)
    axes[0].set_title('Motion Present (No Metasurface)', fontsize=12)
    axes[0].set_ylabel('Phase (normalized)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, preprocess_phase(nomotion_phase), linewidth=0.5, color='blue', alpha=0.8)
    axes[1].set_title('No Motion (No Metasurface)', fontsize=12)
    axes[1].set_ylabel('Phase (normalized)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_phase_strategies_comparison(
    phases_dict: Dict[str, np.ndarray],
    fs: float = 30.0,
    save_path: str = "results/figures/phase_strategies_comparison.png",
):
    """图2: 各策略下相位对比"""
    n = len(phases_dict)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(len(list(phases_dict.values())[0])) / fs

    for ax, (name, phase) in zip(axes, phases_dict.items()):
        p = preprocess_phase(phase)
        ax.plot(t, p, linewidth=0.5, alpha=0.8)
        ax.set_title(f'Strategy: {name}', fontsize=10)
        ax.set_ylabel('Phase')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Phase Comparison Across Strategies (Motion Present)', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_spectrum_strategies_comparison(
    phases_dict: Dict[str, np.ndarray],
    fs: float = 30.0,
    save_path: str = "results/figures/spectrum_strategies_comparison.png",
):
    """图3: 各策略频谱对比"""
    n = len(phases_dict)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (name, phase) in zip(axes, phases_dict.items()):
        p = preprocess_phase(phase)
        freqs, magnitude = compute_spectrum(p, fs)
        mask = freqs <= 5.0
        ax.plot(freqs[mask], magnitude[mask], linewidth=0.8, alpha=0.8)
        ax.axvspan(0.2, 2.5, alpha=0.1, color='red', label='Human motion band')
        ax.set_title(f'Spectrum: {name}', fontsize=10)
        ax.set_ylabel('Magnitude')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frequency (Hz)')
    plt.suptitle('Spectrum Comparison Across Strategies', fontsize=13, y=1.01)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_walking_detection_metrics(
    metrics_dict: Dict[str, Dict],
    save_path: str = "results/figures/walking_detection_metrics.png",
):
    """图4: 运动检测指标对比"""
    strategies = list(metrics_dict.keys())
    n = len(strategies)
    x = np.arange(n)
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Threshold detection
    thresh_acc = [metrics_dict[s].get('threshold_accuracy', 0) for s in strategies]
    thresh_tpr = [metrics_dict[s].get('threshold_tpr', 0) for s in strategies]
    thresh_fpr = [metrics_dict[s].get('threshold_fpr', 0) for s in strategies]

    axes[0].bar(x - width, thresh_acc, width, label='Accuracy', color='steelblue')
    axes[0].bar(x, thresh_tpr, width, label='TPR', color='forestgreen')
    axes[0].bar(x + width, thresh_fpr, width, label='FPR', color='orangered')
    axes[0].set_title('Variance Threshold Detection')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Classifier
    clf_acc = [metrics_dict[s].get('classifier_accuracy', 0) for s in strategies]
    clf_tpr = [metrics_dict[s].get('classifier_tpr', 0) for s in strategies]
    clf_fpr = [metrics_dict[s].get('classifier_fpr', 0) for s in strategies]

    axes[1].bar(x - width, clf_acc, width, label='Accuracy', color='steelblue')
    axes[1].bar(x, clf_tpr, width, label='TPR', color='forestgreen')
    axes[1].bar(x + width, clf_fpr, width, label='FPR', color='orangered')
    axes[1].set_title('Statistical Classifier (LR)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Phase variance & entropy
    variances = [metrics_dict[s].get('mean_phase_variance', 0) for s in strategies]
    entropies = [metrics_dict[s].get('mean_phase_entropy', 0) for s in strategies]

    ax2 = axes[2]
    color1, color2 = 'steelblue', 'orangered'
    ax2.bar(x - width/2, variances, width, label='Phase Variance', color=color1)
    ax2.set_ylabel('Phase Variance', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)

    ax2b = ax2.twinx()
    ax2b.bar(x + width/2, entropies, width, label='Phase Entropy', color=color2)
    ax2b.set_ylabel('Phase Entropy', color=color2)
    ax2b.tick_params(axis='y', labelcolor=color2)

    ax2.set_title('Phase Variance & Entropy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_respiration_error_comparison(
    resp_dict: Dict[str, Dict],
    save_path: str = "results/figures/respiration_error_comparison.png",
):
    """图5: 呼吸频率估计误差对比"""
    strategies = list(resp_dict.keys())
    errors = [resp_dict[s].get('mean_abs_error', 0) for s in strategies]
    stds = [resp_dict[s].get('std_abs_error', 0) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strategies))
    bars = ax.bar(x, errors, yerr=stds, capsize=5, color='steelblue', alpha=0.8,
                  edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Mean Absolute Error (Hz)')
    ax.set_title('Respiration Frequency Estimation Error by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{err:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_summary_comparison(
    metrics_dict: Dict[str, Dict],
    save_path: str = "results/figures/summary_comparison.png",
):
    """图6: 综合对比图"""
    strategies = list(metrics_dict.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Threshold accuracy
    thresh_acc = [metrics_dict[s].get('threshold_accuracy', 0) for s in strategies]
    axes[0, 0].barh(strategies, thresh_acc, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Threshold Detection Accuracy')
    axes[0, 0].set_xlim(0, 1.1)
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # (0,1) Classifier accuracy
    clf_acc = [metrics_dict[s].get('classifier_accuracy', 0) for s in strategies]
    axes[0, 1].barh(strategies, clf_acc, color='forestgreen', alpha=0.8)
    axes[0, 1].set_title('Classifier Accuracy')
    axes[0, 1].set_xlim(0, 1.1)
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # (1,0) Low-freq energy ratio
    lf_ratio = [metrics_dict[s].get('mean_lowfreq_energy_ratio', 0) for s in strategies]
    axes[1, 0].barh(strategies, lf_ratio, color='orangered', alpha=0.8)
    axes[1, 0].set_title('Low-Freq Energy Ratio (0.2-2.5 Hz)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # (1,1) Switching rate
    sw_rate = [metrics_dict[s].get('switching_rate', 0) for s in strategies]
    axes[1, 1].barh(strategies, sw_rate, color='purple', alpha=0.8)
    axes[1, 1].set_title('Switching Rate')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Summary: Privacy Protection vs. Attack Performance', fontsize=14)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_multifreq_frequency_schedule(
    schedule_info: list,
    fs: float = 30.0,
    T: float = 60.0,
    save_path: str = "results/figures/multifreq_frequency_schedule.png",
):
    """图7: multifreq_proposed 子阵列调制频率随时间变化"""
    if not schedule_info:
        print("  [WARNING] No schedule info to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['steelblue', 'forestgreen', 'orangered', 'purple']

    for entry in schedule_info:
        s_idx = entry['subarray_id']
        color = colors[s_idx % len(colors)]
        ax.scatter(entry['time'], entry['new_freq'], color=color, s=20, alpha=0.7,
                   label=f'Subarray {s_idx}' if entry.get('_first', False) else '')

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Multifreq Proposed: Subarray Frequency Schedule')
    ax.axhspan(0.2, 2.5, alpha=0.05, color='red')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_dataset_distribution(
    metadata_df,
    save_path: str = "results/figures/dataset_distribution.png",
):
    """图8: 数据集分布"""
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Strategy distribution
    strategy_counts = metadata_df['strategy'].value_counts()
    axes[0].barh(strategy_counts.index, strategy_counts.values, color='steelblue', alpha=0.8)
    axes[0].set_title('Samples per Strategy')
    axes[0].set_xlabel('Count')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Motion label distribution
    motion_counts = metadata_df['motion_label'].value_counts()
    axes[1].bar(['No Motion (0)', 'Motion (1)'], motion_counts.values,
                color=['blue', 'red'], alpha=0.8)
    axes[1].set_title('Samples per Motion Label')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Split distribution
    split_counts = metadata_df['split'].value_counts()
    axes[2].bar(split_counts.index, split_counts.values, color='forestgreen', alpha=0.8)
    axes[2].set_title('Samples per Split')
    axes[2].set_ylabel('Count')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('RFID-MetaPrivacy-Sim Dataset Distribution', fontsize=13)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_feature_pca(
    phases: np.ndarray,
    labels: np.ndarray,
    strategy_indices: np.ndarray,
    strategies_list: list,
    save_path: str = "results/figures/feature_tsne_or_pca.png",
):
    """图9: PCA 特征分布"""
    from sklearn.decomposition import PCA
    from src.utils import extract_features

    # 提取特征
    X = np.array([extract_features(p) for p in phases])

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 按运动标签着色
    for label, color, name in [(0, 'blue', 'No Motion'), (1, 'red', 'Motion')]:
        mask = labels == label
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.3, s=10, label=name)
    axes[0].set_title('PCA by Motion Label')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 按策略着色
    colors = ['steelblue', 'forestgreen', 'orangered', 'purple', 'goldenrod']
    for s_idx, s_name in enumerate(strategies_list):
        mask = strategy_indices == s_idx
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[s_idx % len(colors)],
                        alpha=0.3, s=10, label=s_name)
    axes[1].set_title('PCA by Strategy')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Feature Space Visualization (PCA)', fontsize=13)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    save_path: str = "results/figures/confusion_matrices.png",
):
    """图10: 混淆矩阵"""
    n = len(confusion_matrices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{name}', fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        # 标注数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=10)

    # 隐藏多余子图
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices (Classifier)', fontsize=13)
    plt.tight_layout()
    _save_fig(fig, save_path)
