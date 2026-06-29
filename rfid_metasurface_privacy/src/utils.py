"""
工具函数：相位预处理、频谱分析、特征提取
不依赖 torch
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple


def make_rng(seed=None):
    """创建 numpy.random.Generator"""
    return np.random.default_rng(seed)


def preprocess_phase(phase: np.ndarray, detrend: bool = True,
                     zero_mean: bool = True, standardize: bool = False) -> np.ndarray:
    """
    相位预处理流水线：
    1. unwrap
    2. detrend (去除线性趋势)
    3. zero-mean
    4. standardize (可选)
    """
    p = np.unwrap(phase)
    if detrend:
        p = sp_signal.detrend(p, type='linear')
    if zero_mean:
        p = p - np.mean(p)
    if standardize:
        std = np.std(p)
        if std > 1e-12:
            p = p / std
    return p


def compute_spectrum(phase: np.ndarray, fs: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单边幅度谱

    Returns:
        freqs: 频率数组
        magnitude: 幅度数组
    """
    n = len(phase)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(phase)
    magnitude = np.abs(spectrum) * 2.0 / n
    return freqs, magnitude


def compute_spectral_entropy(phase: np.ndarray, fs: float = 30.0) -> float:
    """计算频谱熵"""
    freqs, magnitude = compute_spectrum(phase, fs)
    power = magnitude ** 2
    total = np.sum(power)
    if total < 1e-20:
        return 0.0
    prob = power / total
    prob = prob[prob > 1e-20]
    return float(-np.sum(prob * np.log2(prob)))


def compute_phase_entropy(phase: np.ndarray, n_bins: int = 64) -> float:
    """
    计算相位直方图熵

    对 detrend + zero_mean + standardize 后的 phase 计算
    histogram bins = 64, range = [-3, 3]
    """
    p = phase - np.mean(phase)
    std = np.std(p)
    if std < 1e-12:
        return 0.0
    p = p / std  # standardize
    hist, _ = np.histogram(p, bins=n_bins, range=(-3, 3), density=False)
    hist = hist.astype(float)
    total = hist.sum()
    if total < 1e-10:
        return 0.0
    prob = hist / total
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_lowfreq_energy_ratio(phase: np.ndarray, fs: float = 30.0,
                                  low_min: float = 0.2, low_max: float = 2.5) -> float:
    """计算低频段能量占比"""
    freqs, magnitude = compute_spectrum(phase, fs)
    power = magnitude ** 2
    total = np.sum(power)
    if total < 1e-20:
        return 0.0
    mask = (freqs >= low_min) & (freqs <= low_max)
    return float(np.sum(power[mask]) / total)


def compute_band_energy_ratio(phase: np.ndarray, fs: float = 30.0,
                               fmin: float = 0.2, fmax: float = 0.5) -> float:
    """计算指定频段能量占比"""
    freqs, magnitude = compute_spectrum(phase, fs)
    power = magnitude ** 2
    total = np.sum(power)
    if total < 1e-20:
        return 0.0
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(power[mask]) / total)


def compute_dominant_frequency(phase: np.ndarray, fs: float = 30.0,
                                fmin: float = 0.1, fmax: float = 5.0) -> float:
    """计算主频率"""
    freqs, magnitude = compute_spectrum(phase, fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(magnitude[mask])
    return float(freqs[mask][idx])


def compute_autocorrelation_peak(phase: np.ndarray, max_lag: int = 150) -> float:
    """计算自相关峰值（归一化）"""
    p = phase - np.mean(phase)
    n = len(p)
    max_lag = min(max_lag, n // 2)
    var = np.var(p)
    if var < 1e-20:
        return 0.0
    acf = np.correlate(p[:max_lag * 2], p[:max_lag * 2], mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / (var * len(p))
    if len(acf) > 10:
        return float(np.max(acf[5:]))
    return 0.0


def extract_features(phase: np.ndarray, fs: float = 30.0) -> np.ndarray:
    """
    提取统计特征向量（12维）
    """
    p = preprocess_phase(phase, detrend=True, zero_mean=True, standardize=False)
    features = [
        np.mean(p),
        np.std(p),
        np.var(p),
        np.ptp(p),
        np.sqrt(np.mean(p ** 2)),
        compute_dominant_frequency(p, fs),
        compute_lowfreq_energy_ratio(p, fs),
        compute_spectral_entropy(p, fs),
        compute_phase_entropy(p),
        compute_autocorrelation_peak(p),
        compute_band_energy_ratio(p, fs, 0.2, 0.5),
        compute_band_energy_ratio(p, fs, 0.5, 2.5),
    ]
    return np.array(features, dtype=np.float64)


FEATURE_NAMES = [
    'mean', 'std', 'variance', 'peak_to_peak', 'rms',
    'dominant_freq', 'lowfreq_energy_ratio', 'spectral_entropy',
    'phase_entropy', 'autocorrelation_peak',
    'band_energy_0.2_0.5', 'band_energy_0.5_2.5',
]
