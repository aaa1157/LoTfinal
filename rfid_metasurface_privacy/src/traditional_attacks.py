"""
传统攻击者

攻击1: 方差阈值检测
攻击2: 统计特征分类器 (Logistic Regression)
攻击3: 呼吸频率估计
攻击4: 周期干扰可滤除性分析

不依赖 torch
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Tuple

from src.utils import (
    preprocess_phase, extract_features, compute_spectrum,
    FEATURE_NAMES,
)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算分类指标"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))

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


def variance_threshold_detection(
    motion_phases: List[np.ndarray],
    nomotion_phases: List[np.ndarray],
) -> Dict[str, float]:
    """
    攻击1: 方差阈值检测

    1. 计算 no_motion 相位方差
    2. threshold = mean(var_no_motion) + 3 * std(var_no_motion)
    3. 判断 motion 样本是否超过阈值
    """
    nomotion_processed = [preprocess_phase(p) for p in nomotion_phases]
    motion_processed = [preprocess_phase(p) for p in motion_phases]

    nomotion_vars = [np.var(p) for p in nomotion_processed]
    mean_var = np.mean(nomotion_vars)
    std_var = np.std(nomotion_vars)
    threshold = mean_var + 3 * std_var

    motion_vars = [np.var(p) for p in motion_processed]

    y_true = np.array([1] * len(motion_vars) + [0] * len(nomotion_vars))
    y_pred = np.array(
        [1 if v > threshold else 0 for v in motion_vars] +
        [1 if v > threshold else 0 for v in nomotion_vars]
    )

    metrics = _compute_metrics(y_true, y_pred)
    metrics['threshold'] = float(threshold)
    return metrics


def statistical_classifier(
    motion_phases: List[np.ndarray],
    nomotion_phases: List[np.ndarray],
    fs: float = 30.0,
) -> Dict[str, float]:
    """
    攻击2: 统计特征分类器 (Logistic Regression)

    特征: 12维统计特征
    """
    X_motion = np.array([extract_features(p, fs) for p in motion_phases])
    X_nomotion = np.array([extract_features(p, fs) for p in nomotion_phases])

    y_motion = np.ones(len(X_motion))
    y_nomotion = np.zeros(len(X_nomotion))

    X = np.vstack([X_motion, X_nomotion])
    y = np.concatenate([y_motion, y_nomotion])

    # 打乱
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    # 训练 Logistic Regression
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(X, y)
    y_pred = clf.predict(X)

    metrics = _compute_metrics(y, y_pred)

    # 保存特征系数
    lr = clf.named_steps['lr']
    metrics['feature_coefficients'] = dict(zip(FEATURE_NAMES, lr.coef_[0]))

    return metrics


def respiration_estimation(
    phases: List[np.ndarray],
    true_freq: float,
    fs: float = 30.0,
) -> Dict[str, float]:
    """
    攻击3: 呼吸频率估计

    1. detrend
    2. 在 0.2-0.5 Hz 范围内做 FFT
    3. 找最大谱峰
    4. 与 true_freq 比较
    """
    estimated_freqs = []

    for phase in phases:
        p = preprocess_phase(phase)
        freqs, magnitude = compute_spectrum(p, fs)

        mask = (freqs >= 0.2) & (freqs <= 0.5)
        if np.any(mask):
            idx = np.argmax(magnitude[mask])
            est_freq = freqs[mask][idx]
        else:
            est_freq = 0.0
        estimated_freqs.append(est_freq)

    estimated_freqs = np.array(estimated_freqs)
    abs_errors = np.abs(estimated_freqs - true_freq)

    return {
        'true_respiration_freq': float(true_freq),
        'mean_estimated_freq': float(np.mean(estimated_freqs)),
        'mean_abs_error': float(np.mean(abs_errors)),
        'std_abs_error': float(np.std(abs_errors)),
    }


def periodic_filtering_analysis(
    motion_phases: List[np.ndarray],
    nomotion_phases: List[np.ndarray],
    fs: float = 30.0,
    notch_freq: float = 2.0,
    notch_width: float = 0.3,
) -> Dict[str, float]:
    """
    攻击4: 周期干扰可滤除性分析

    针对 periodic 策略：
    1. 检测频谱尖峰
    2. notch-like 去除该频率附近分量
    3. 比较滤波前后分类器效果
    """
    from scipy.signal import butter, filtfilt

    def notch_filter(phase, freq, width, fs):
        """简单的频域 notch 滤波"""
        n = len(phase)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        spectrum = np.fft.rfft(phase)
        mask = np.abs(freqs - freq) > width
        spectrum[~mask] = 0
        return np.fft.irfft(spectrum, n=n)

    # 滤波前分类
    pre_metrics = statistical_classifier(motion_phases, nomotion_phases, fs)

    # 滤波后
    motion_filtered = [notch_filter(preprocess_phase(p), notch_freq, notch_width, fs)
                       for p in motion_phases]
    nomotion_filtered = [notch_filter(preprocess_phase(p), notch_freq, notch_width, fs)
                         for p in nomotion_phases]

    post_metrics = statistical_classifier(motion_filtered, nomotion_filtered, fs)

    return {
        'pre_filter_accuracy': pre_metrics['accuracy'],
        'post_filter_accuracy': post_metrics['accuracy'],
        'accuracy_drop': pre_metrics['accuracy'] - post_metrics['accuracy'],
        'pre_filter_tpr': pre_metrics['tpr'],
        'post_filter_tpr': post_metrics['tpr'],
    }
