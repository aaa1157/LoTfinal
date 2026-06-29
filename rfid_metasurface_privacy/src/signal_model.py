"""
RFID 机理级仿真信号模型

R(t) = v_static + v_human(t) + v_meta(t) + noise

攻击者观测量: phi(t) = unwrap(angle(R(t)))

不依赖 torch
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List

from src.config import (
    fs as DEFAULT_FS, T as DEFAULT_T, sequence_length as DEFAULT_LENGTH,
    motion_freq_range, noise_std_default, wall_types, snr_db_choices,
    motion_types, motion_type_freq_ranges, motion_type_amp_ranges,
)
from src.utils import make_rng, preprocess_phase


def make_time_axis(fs: float = DEFAULT_FS, T: float = DEFAULT_T) -> np.ndarray:
    """返回时间轴 t"""
    return np.arange(int(fs * T)) / fs


def generate_static_vector(
    rng: np.random.Generator,
    n_paths: int = 4,
    amplitude_range: Tuple[float, float] = (0.3, 1.0),
) -> complex:
    """
    生成静态复向量 v_static
    包括 LoS 和环境静态反射，多个路径叠加
    """
    total = 0.0 + 0.0j
    for i in range(n_paths):
        amp = rng.uniform(amplitude_range[0], amplitude_range[1]) / n_paths
        phase = rng.uniform(0, 2 * np.pi)
        total += amp * np.exp(1j * phase)
    return total


def generate_walking_human_reflection(
    t: np.ndarray,
    rng: np.random.Generator,
    motion: bool = True,
    motion_type: str = "walking_normal",
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    生成人体走动动态反射 v_human(t)

    motion=True: 根据运动类型生成对应频率范围的扰动
    motion=False: 只有静态环境微扰和噪声

    Returns:
        v_human: shape=(len(t),), complex
        true_motion_freqs: 运动频率分量列表
    """
    length = len(t)
    v_human = np.zeros(length, dtype=complex)
    true_freqs = []

    if params is None:
        params = {}

    if motion:
        # 根据运动类型确定频率和幅度范围
        freq_range = motion_type_freq_ranges.get(motion_type, motion_freq_range)
        amp_range = motion_type_amp_ranges.get(motion_type, (0.08, 0.25))

        # 多个正弦分量模拟躯干和四肢运动
        n_components = rng.integers(3, 7)
        human_amp_base = params.get('human_amplitude', rng.uniform(amp_range[0], amp_range[1]))

        for _ in range(n_components):
            freq = rng.uniform(freq_range[0], freq_range[1])
            amp = human_amp_base * rng.uniform(0.3, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            true_freqs.append(float(freq))
            v_human += amp * np.exp(1j * (2 * np.pi * freq * t + phase))

        # 轻微非平稳扰动：缓慢幅度调制
        mod_freq = rng.uniform(0.02, 0.1)
        mod_amp = rng.uniform(0.01, 0.05)
        modulation = 1.0 + mod_amp * np.sin(2 * np.pi * mod_freq * t)
        v_human = v_human * modulation

        # 轻微频率漂移
        drift_rate = rng.uniform(-0.01, 0.01)
        drift = np.exp(1j * 2 * np.pi * drift_rate * t ** 2)
        v_human = v_human * drift

        # 轻微随机噪声
        noise_amp = rng.uniform(0.005, 0.02)
        v_human += noise_amp * (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2)
    else:
        # 无人体运动：只有静态环境微扰
        # 让 no_motion 也有环境微扰，避免过于理想化
        micro_amp = params.get('environmental_micro_motion_strength',
                               rng.uniform(0.005, 0.02))
        # 低频环境微扰（空调振动、建筑微震等）
        n_micro = rng.integers(1, 3)
        for _ in range(n_micro):
            micro_freq = rng.uniform(0.01, 0.15)
            micro_phase = rng.uniform(0, 2 * np.pi)
            v_human += micro_amp * 0.5 * np.exp(1j * (2 * np.pi * micro_freq * t + micro_phase))
        # 随机微扰
        v_human += micro_amp * 0.3 * (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2)

    return v_human, true_freqs


def generate_respiration_reflection(
    t: np.ndarray,
    rng: np.random.Generator,
    respiration_freq: float = 0.3,
    params: Optional[Dict] = None,
) -> Tuple[np.ndarray, float]:
    """
    生成呼吸动态反射

    Returns:
        v_resp: shape=(len(t),), complex
        true_respiration_freq: 实际呼吸频率
    """
    if params is None:
        params = {}

    amp = params.get('resp_amplitude', rng.uniform(0.02, 0.06))
    phase = rng.uniform(0, 2 * np.pi)
    # 呼吸引起的相位调制
    resp_phase = amp * np.sin(2 * np.pi * respiration_freq * t + phase)
    v_resp = np.exp(1j * resp_phase) - 1.0  # 小信号近似

    return v_resp, respiration_freq


def generate_complex_noise(
    t: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = noise_std_default,
) -> np.ndarray:
    """生成复高斯噪声"""
    length = len(t)
    return noise_std * (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2)


def snr_to_noise_std(signal_power: float, snr_db: float) -> float:
    """根据 SNR (dB) 计算噪声标准差"""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    return float(np.sqrt(noise_power / 2))


def simulate_received_signal(
    t: np.ndarray,
    strategy_name: str = "no_metasurface",
    motion_label: int = 1,
    task_type: str = "walking_detection",
    rng: Optional[np.random.Generator] = None,
    params: Optional[Dict] = None,
    meta_component: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    统一仿真入口

    Args:
        t: 时间轴
        strategy_name: 超表面策略名
        motion_label: 0=no_motion, 1=motion
        task_type: "walking_detection" or "respiration_estimation"
        rng: 随机数生成器
        params: 仿真参数
        meta_component: 超表面信号分量

    Returns:
        complex_signal: R(t)
        observed_phase: phi(t)
        metadata: dict
    """
    if rng is None:
        rng = make_rng()
    if params is None:
        params = {}

    length = len(t)
    motion = (motion_label == 1)
    noise_std = params.get('noise_std', noise_std_default)

    # 运动类型
    motion_type = params.get('motion_type', 'walking_normal')

    # 墙体衰减
    wall_type = params.get('wall_type', 'wooden')
    wall_atten = wall_types.get(wall_type, 0.15)

    # 静态分量
    static_amp_range = params.get('static_path_amplitude_range', (0.3, 1.0))
    v_static = generate_static_vector(rng, amplitude_range=static_amp_range)
    v_static *= (1.0 - wall_atten)

    # 人体运动分量
    v_human, true_motion_freqs = generate_walking_human_reflection(
        t, rng, motion=motion, motion_type=motion_type, params=params
    )
    v_human *= (1.0 - wall_atten)

    # 呼吸分量
    v_resp = np.zeros(length, dtype=complex)
    true_resp_freq = float('nan')
    if task_type == "respiration_estimation":
        resp_freq = params.get('respiration_freq', rng.choice([0.25, 0.30]))
        v_resp, true_resp_freq = generate_respiration_reflection(
            t, rng, respiration_freq=resp_freq, params=params
        )
        v_resp *= (1.0 - wall_atten)

    # 超表面分量
    v_meta = np.zeros(length, dtype=complex)
    if meta_component is not None:
        v_meta = meta_component

    # 噪声
    noise = generate_complex_noise(t, rng, noise_std)

    # 合成信号
    R = v_static + v_human + v_resp + v_meta + noise

    # 提取相位
    phi = np.unwrap(np.angle(R))

    metadata = {
        'strategy': strategy_name,
        'motion_label': motion_label,
        'task_type': task_type,
        'motion_type': motion_type if motion else 'none',
        'true_motion_freqs': true_motion_freqs,
        'true_respiration_freq': true_resp_freq,
        'noise_std': noise_std,
        'wall_type': wall_type,
        'wall_attenuation': wall_atten,
        'human_amplitude': params.get('human_amplitude', 0.15),
        'metasurface_amplitude': params.get('metasurface_amplitude', 0.08),
        'snr_db': params.get('snr_db', 20),
        'scene_id': params.get('scene_id', 0),
        'environment_id': params.get('environment_id', 0),
    }

    return R, phi, metadata
