"""
隐私评估指标

不依赖 torch
"""

import numpy as np
from typing import List, Dict

from src.utils import (
    preprocess_phase, compute_spectral_entropy, compute_phase_entropy,
    compute_lowfreq_energy_ratio, compute_dominant_frequency,
)
from src.metasurface import compute_switching_rate


def compute_phase_variance(phases: List[np.ndarray]) -> float:
    """计算平均相位方差"""
    variances = [np.var(preprocess_phase(p)) for p in phases]
    return float(np.mean(variances))


def compute_avg_phase_entropy(phases: List[np.ndarray]) -> float:
    """计算平均相位熵"""
    entropies = [compute_phase_entropy(preprocess_phase(p)) for p in phases]
    return float(np.mean(entropies))


def compute_avg_spectral_entropy(phases: List[np.ndarray], fs: float = 30.0) -> float:
    """计算平均频谱熵"""
    entropies = [compute_spectral_entropy(preprocess_phase(p), fs) for p in phases]
    return float(np.mean(entropies))


def compute_avg_lowfreq_energy_ratio(phases: List[np.ndarray], fs: float = 30.0) -> float:
    """计算平均低频能量比"""
    ratios = [compute_lowfreq_energy_ratio(preprocess_phase(p), fs) for p in phases]
    return float(np.mean(ratios))


def compute_avg_switching_rate(states_list: List[np.ndarray], fs: float = 30.0) -> float:
    """计算平均切换率"""
    rates = [compute_switching_rate(s, fs) for s in states_list]
    return float(np.mean(rates)) if rates else 0.0


def compute_privacy_gain_accuracy(baseline_acc: float, defended_acc: float) -> float:
    """计算隐私增益（准确率下降）"""
    return float(baseline_acc - defended_acc)


def compute_privacy_gain_respiration(
    baseline_error: float, defended_error: float
) -> float:
    """计算隐私增益（呼吸误差提升）"""
    if baseline_error < 1e-10:
        return 0.0
    return float(defended_error / baseline_error)
