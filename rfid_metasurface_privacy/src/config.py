"""
全局配置文件 - 集中管理默认参数
"""

RANDOM_SEED = 2026

# 信号参数
fs = 30
T = 60
sequence_length = 1800

# 超表面参数
metasurface_units = 16
metasurface_shape = (4, 4)

# 策略列表
strategies = [
    "no_metasurface",
    "periodic",
    "random",
    "rfnoid_like",
    "multifreq_proposed",
]

# 频段定义
motion_freq_range = (0.5, 2.5)
human_motion_band = (0.2, 2.5)
respiration_band = (0.2, 0.5)

# 呼吸频率
default_respiration_freqs = [0.25, 0.30]

# 运动类型
motion_types = [
    "walking_slow",
    "walking_normal",
    "walking_fast",
    "arm_swing",
    "random_body_motion",
]

# 运动类型对应的频率范围
motion_type_freq_ranges = {
    "walking_slow": (0.5, 1.0),
    "walking_normal": (0.8, 1.8),
    "walking_fast": (1.2, 2.5),
    "arm_swing": (0.5, 1.5),
    "random_body_motion": (0.3, 2.5),
}

# 运动类型对应的幅度范围
motion_type_amp_ranges = {
    "walking_slow": (0.05, 0.15),
    "walking_normal": (0.08, 0.25),
    "walking_fast": (0.12, 0.35),
    "arm_swing": (0.04, 0.15),
    "random_body_motion": (0.06, 0.30),
}

# SNR 集合 (dB)
snr_db_choices = [-5, 0, 5, 10, 15, 20, 25, 30]

# 数据集规模 (按 mode)
dataset_modes = {
    "debug": {
        "walk_samples_per_strategy_per_label": 100,
        "resp_samples_per_strategy": 50,
    },
    "medium": {
        "walk_samples_per_strategy_per_label": 500,
        "resp_samples_per_strategy": 200,
    },
    "full": {
        "walk_samples_per_strategy_per_label": 1000,
        "resp_samples_per_strategy": 500,
    },
}

# 兼容旧参数
dataset_full_samples_per_strategy_per_label = 1000
dataset_fast_samples_per_strategy_per_label = 100
dataset_full_resp_samples_per_strategy = 500
dataset_fast_resp_samples_per_strategy = 50

# 数据集划分
train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

# 噪声参数
noise_std_default = 0.03
snr_db_default = 20

# 墙体类型及衰减
wall_types = {
    "glass": 0.05,
    "wooden": 0.15,
    "gypsum": 0.25,
    "concrete_light": 0.40,
}

# 深度学习训练 (按 mode)
dl_modes = {
    "debug": {
        "epochs": 10,
        "patience": 3,
        "batch_size": 64,
    },
    "medium": {
        "epochs": 50,
        "patience": 8,
        "batch_size": 128,
    },
    "full": {
        "epochs": 100,
        "patience": 12,
        "batch_size": 128,
    },
}

dl_learning_rate = 1e-3
dl_weight_decay = 1e-4

# LMC 控制器 (按 mode)
lmc_modes = {
    "debug": {
        "search_iters": 50,
        "eval_seeds": 1,
    },
    "medium": {
        "search_iters": 200,
        "eval_seeds": 3,
    },
    "full": {
        "search_iters": 500,
        "eval_seeds": 5,
    },
}

lmc_alpha = 0.5    # respiration error weight
lmc_beta = 0.3     # spectral entropy weight
lmc_eta = 0.2      # lowfreq energy ratio weight
lmc_gamma = 0.1    # switching cost weight
