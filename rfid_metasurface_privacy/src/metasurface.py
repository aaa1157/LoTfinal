"""
1-bit 可编程超表面模型和控制策略

超表面有 N=16 个 1-bit 单元，对应 4×4 metasurface
每个单元状态 s_i(t) ∈ {0, 1}，对应反射相位 0 或 pi

v_meta(t) = sum_i A_i * exp(-j * (base_phase_i + pi * s_i(t)))

5种策略 + LMC控制器策略

不依赖 torch
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from src.config import metasurface_units, metasurface_shape, human_motion_band
from src.utils import make_rng


def initialize_metasurface_params(
    N: int = metasurface_units,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    初始化超表面参数

    Returns:
        amplitudes: (N,) 每个单元的反射幅度
        base_phases: (N,) 每个单元的基础相位
    """
    if rng is None:
        rng = make_rng()
    amplitudes = rng.uniform(0.06, 0.10, size=N)
    base_phases = rng.uniform(0, 2 * np.pi, size=N)
    return amplitudes, base_phases


def compute_metasurface_reflection(
    states: np.ndarray,
    amplitudes: np.ndarray,
    base_phases: np.ndarray,
) -> np.ndarray:
    """
    计算超表面反射信号

    Args:
        states: (T, N) 0/1 状态矩阵
        amplitudes: (N,) 反射幅度
        base_phases: (N,) 基础相位

    Returns:
        v_meta: (T,) 复数信号
    """
    # phases: (T, N)
    phases = base_phases[np.newaxis, :] + np.pi * states
    # v_meta: (T,)
    v_meta = np.sum(amplitudes[np.newaxis, :] * np.exp(-1j * phases), axis=1)
    return v_meta


def compute_switching_rate(states: np.ndarray, fs: float = 30.0) -> float:
    """
    计算超表面状态切换率

    Args:
        states: (T, N) 0/1 状态矩阵

    Returns:
        switching_rate: 每秒每单元平均切换次数
    """
    if states.shape[0] <= 1:
        return 0.0
    diffs = np.abs(np.diff(states, axis=0))
    total_switches = np.sum(diffs)
    duration = (states.shape[0] - 1) / fs
    n_elements = states.shape[1]
    return float(total_switches / duration / n_elements)


def get_subarray_groups(shape: Tuple[int, int] = metasurface_shape) -> List[List[int]]:
    """
    返回四个 2×2 子阵列分组

    对于 4×4 超表面：
    子阵列0: 左上 2×2
    子阵列1: 右上 2×2
    子阵列2: 左下 2×2
    子阵列3: 右下 2×2
    """
    rows, cols = shape
    groups = []
    for r in range(0, rows, 2):
        for c in range(0, cols, 2):
            group = []
            for dr in range(2):
                for dc in range(2):
                    idx = (r + dr) * cols + (c + dc)
                    group.append(idx)
            groups.append(group)
    return groups


def generate_state_schedule(
    strategy: str,
    t: np.ndarray,
    N: int = metasurface_units,
    rng: Optional[np.random.Generator] = None,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, Optional[List[Dict]]]:
    """
    生成超表面状态调度

    Args:
        strategy: 策略名
        t: 时间轴
        N: 单元数
        rng: 随机数生成器
        params: 策略参数

    Returns:
        states: (T, N) 0/1 状态矩阵
        schedule_info: 调度信息（用于画图等）
    """
    if rng is None:
        rng = make_rng()
    if params is None:
        params = {}

    length = len(t)
    fs_val = 1.0 / (t[1] - t[0]) if length > 1 else 30.0

    if strategy == "no_metasurface":
        states = np.zeros((length, N), dtype=np.int8)
        return states, None

    elif strategy == "periodic":
        flip_freq = params.get('flip_freq', 2.0)
        states = np.zeros((length, N), dtype=np.int8)
        flip_signal = (np.sin(2 * np.pi * flip_freq * t) > 0).astype(np.int8)
        states[:, :] = flip_signal[:, np.newaxis]
        return states, None

    elif strategy == "random":
        flip_prob = params.get('flip_prob', 0.3)
        states = np.zeros((length, N), dtype=np.int8)
        prev = np.zeros(N, dtype=np.int8)
        for i in range(length):
            flip_mask = rng.random(N) < flip_prob
            prev = (prev + flip_mask.astype(np.int8)) % 2
            states[i] = prev.copy()
        return states, None

    elif strategy == "rfnoid_like":
        small_ratio = params.get('small_ratio', 0.25)
        large_ratio = params.get('large_ratio', 0.75)
        lowfreq_weight = params.get('lowfreq_weight', 0.6)

        states = np.zeros((length, N), dtype=np.int8)
        prev = np.zeros(N, dtype=np.int8)

        for i in range(length):
            r = rng.random()
            if r < 0.15:
                # 大规模翻转
                n_flip = rng.integers(int(N * large_ratio), N + 1)
            elif r < 0.15 + lowfreq_weight * 0.4:
                # 小规模翻转（增强低频）
                n_flip = rng.integers(1, max(2, int(N * small_ratio)))
            else:
                # 中等翻转
                n_flip = rng.integers(1, int(N * 0.5) + 1)

            n_flip = min(n_flip, N)
            flip_indices = rng.choice(N, size=n_flip, replace=False)
            flip_mask = np.zeros(N, dtype=bool)
            flip_mask[flip_indices] = True
            prev = (prev + flip_mask.astype(np.int8)) % 2
            states[i] = prev.copy()

        return states, None

    elif strategy == "multifreq_proposed":
        subarrays = get_subarray_groups()
        n_sub = len(subarrays)
        init_freqs = params.get('init_freqs', [0.3, 0.8, 1.5, 2.3])
        random_flip_prob = params.get('random_flip_prob', 0.1)
        regroup_prob = params.get('regroup_prob', 0.0)
        small_ratio = params.get('small_ratio', 0.25)
        large_ratio = params.get('large_ratio', 0.75)
        lowfreq_weight = params.get('lowfreq_weight', 0.6)
        hop_interval_range = params.get('hop_interval_range', (3, 8))
        metasurface_amplitude = params.get('metasurface_amplitude', None)
        switching_penalty = params.get('switching_penalty', 0.0)

        states = np.zeros((length, N), dtype=np.int8)
        sub_freqs = list(init_freqs[:n_sub])
        sub_phases = rng.uniform(0, 2 * np.pi, n_sub)
        hop_timers = rng.uniform(hop_interval_range[0], hop_interval_range[1], n_sub)
        hop_counters = np.zeros(n_sub)

        # 频率调度信息
        schedule_info = []
        current_freqs = list(sub_freqs)

        for i in range(length):
            current_time = t[i]

            for s_idx in range(n_sub):
                # 跳频机制
                hop_counters[s_idx] += 1.0 / fs_val
                if hop_counters[s_idx] >= hop_timers[s_idx]:
                    old_freq = sub_freqs[s_idx]
                    sub_freqs[s_idx] = rng.uniform(human_motion_band[0], human_motion_band[1])
                    sub_phases[s_idx] = rng.uniform(0, 2 * np.pi)
                    hop_timers[s_idx] = rng.uniform(hop_interval_range[0], hop_interval_range[1])
                    hop_counters[s_idx] = 0.0
                    schedule_info.append({
                        'time': current_time,
                        'subarray_id': s_idx,
                        'old_freq': old_freq,
                        'new_freq': sub_freqs[s_idx],
                    })

                # 子阵列调制
                base_state = (np.sin(2 * np.pi * sub_freqs[s_idx] * current_time
                                     + sub_phases[s_idx]) > 0).astype(np.int8)

                for elem_idx in subarrays[s_idx]:
                    if rng.random() < random_flip_prob:
                        states[i, elem_idx] = 1 - base_state
                    else:
                        states[i, elem_idx] = base_state

            # 随机重分组（低概率）
            if regroup_prob > 0 and rng.random() < regroup_prob:
                perm = rng.permutation(N)
                for s_idx in range(n_sub):
                    subarrays[s_idx] = list(perm[s_idx * (N // n_sub):(s_idx + 1) * (N // n_sub)])

        return states, schedule_info

    elif strategy == "lmc_controller":
        # LMC 控制器策略 - 参数由外部传入
        lmc_params = params.get('lmc_params', None)
        if lmc_params is not None:
            # 使用 LMC 参数生成 multifreq 风格的调度
            lmc_params['strategy'] = 'multifreq_proposed'
            return generate_state_schedule('multifreq_proposed', t, N, rng, lmc_params)
        else:
            # 默认退回 multifreq
            return generate_state_schedule('multifreq_proposed', t, N, rng, params)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_metasurface_signal(
    strategy: str,
    t: np.ndarray,
    N: int = metasurface_units,
    rng: Optional[np.random.Generator] = None,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """
    生成超表面信号分量

    Returns:
        v_meta: (T,) 复数信号
        states: (T, N) 0/1 状态矩阵
        schedule_info: 调度信息
    """
    if rng is None:
        rng = make_rng()
    if params is None:
        params = {}

    ms_rng = make_rng(rng.integers(0, 2**31))
    param_rng = make_rng(rng.integers(0, 2**31))

    amplitudes, base_phases = initialize_metasurface_params(N, param_rng)

    # 如果指定了 metasurface_amplitude，覆盖默认幅度
    meta_amp = params.get('metasurface_amplitude', None)
    if meta_amp is not None:
        amplitudes = np.full(N, meta_amp)

    states, schedule_info = generate_state_schedule(strategy, t, N, ms_rng, params)

    if strategy == "no_metasurface":
        v_meta = np.zeros(len(t), dtype=complex)
    else:
        v_meta = compute_metasurface_reflection(states, amplitudes, base_phases)

    return v_meta, states, schedule_info
