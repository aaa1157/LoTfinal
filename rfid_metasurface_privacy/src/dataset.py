"""
RFID-MetaPrivacy-Sim 数据集构建模块

支持 mode: debug / medium / full
支持 split: random / scene_disjoint

不依赖 torch
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List

from src.config import (
    RANDOM_SEED, fs as DEFAULT_FS, T as DEFAULT_T, sequence_length as DEFAULT_LENGTH,
    strategies, default_respiration_freqs, noise_std_default, snr_db_default,
    dataset_modes, train_ratio, val_ratio, test_ratio, wall_types,
    motion_types, snr_db_choices,
)
from src.utils import make_rng, preprocess_phase, extract_features
from src.signal_model import make_time_axis, simulate_received_signal
from src.metasurface import generate_metasurface_signal


def build_dataset(
    mode: str = "debug",
    split_type: str = "random",
    force: bool = False,
    seed: int = RANDOM_SEED,
    base_dir: str = None,
) -> Dict:
    """
    构建完整数据集

    Args:
        mode: debug / medium / full
        split_type: random / scene_disjoint
        force: 强制重新生成
        seed: 随机种子
        base_dir: 项目根目录

    Returns:
        dict with keys: X_phase, y_motion, strategy_index, task_index, metadata_df
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    processed_dir = os.path.join(base_dir, 'data', 'processed')
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    suffix = f"_{mode}_{split_type}"
    npz_path = os.path.join(processed_dir, f"rfid_metaprivacy_sim{suffix}.npz")
    csv_path = os.path.join(processed_dir, f"rfid_metaprivacy_metadata{suffix}.csv")

    if not force and os.path.exists(npz_path) and os.path.exists(csv_path):
        print(f"  Dataset already exists: {npz_path}")
        print(f"  Use --force to rebuild.")
        return load_dataset(mode=mode, split_type=split_type, base_dir=base_dir)

    mode_config = dataset_modes.get(mode, dataset_modes["debug"])
    n_walk = mode_config["walk_samples_per_strategy_per_label"]
    n_resp = mode_config["resp_samples_per_strategy"]

    rng = make_rng(seed)
    t = make_time_axis()

    all_phases = []
    all_y_motion = []
    all_strategy_idx = []
    all_task_idx = []
    all_metadata = []

    strategy_list = list(strategies)
    wall_type_list = list(wall_types.keys())

    # Scene IDs: 0-19 for train/val, 20-29 for test (scene_disjoint)
    n_scenes = 30

    sample_id = 0

    for s_idx, strategy_name in enumerate(strategy_list):
        print(f"  Building walking samples: {strategy_name} ...")

        for motion_label in [0, 1]:
            for i in range(n_walk):
                sample_rng = make_rng(rng.integers(0, 2**31))

                # 随机化参数
                wall_type = wall_type_list[sample_rng.integers(0, len(wall_type_list))]
                scene_id = int(sample_rng.integers(0, n_scenes))
                env_id = int(sample_rng.integers(0, 5))
                snr_db = float(sample_rng.choice(snr_db_choices))
                noise_std = 0.01 + 0.04 * (1.0 - (snr_db + 5) / 35.0)  # SNR -> noise_std
                noise_std = max(0.005, min(noise_std, 0.1))
                human_amp = sample_rng.uniform(0.08, 0.25)
                motion_type_val = str(sample_rng.choice(motion_types)) if motion_label == 1 else 'none'
                micro_strength = sample_rng.uniform(0.005, 0.02)

                params = {
                    'noise_std': noise_std,
                    'wall_type': wall_type,
                    'human_amplitude': human_amp,
                    'snr_db': snr_db,
                    'scene_id': scene_id,
                    'environment_id': env_id,
                    'motion_type': motion_type_val,
                    'environmental_micro_motion_strength': micro_strength,
                }

                meta_rng = make_rng(sample_rng.integers(0, 2**31))
                v_meta, _, _ = generate_metasurface_signal(strategy_name, t, rng=meta_rng)

                _, phi, meta = simulate_received_signal(
                    t, strategy_name=strategy_name,
                    motion_label=motion_label,
                    task_type="walking_detection",
                    rng=sample_rng,
                    params=params,
                    meta_component=v_meta,
                )

                all_phases.append(phi)
                all_y_motion.append(motion_label)
                all_strategy_idx.append(s_idx)
                all_task_idx.append(0)

                meta_record = {
                    'sample_id': sample_id,
                    'strategy': strategy_name,
                    'motion_label': motion_label,
                    'task_type': 'walking_detection',
                    'motion_type': motion_type_val,
                    'true_motion_freqs': str(meta.get('true_motion_freqs', [])),
                    'true_respiration_freq': meta.get('true_respiration_freq', float('nan')),
                    'snr_db': snr_db,
                    'human_amplitude': human_amp,
                    'metasurface_amplitude': meta.get('metasurface_amplitude', 0.08),
                    'noise_std': noise_std,
                    'seed': int(sample_rng.integers(0, 2**31)),
                    'split': '',
                    'scene_id': scene_id,
                    'wall_type': wall_type,
                    'environment_id': env_id,
                }
                all_metadata.append(meta_record)
                sample_id += 1

        # Respiration
        print(f"  Building respiration samples: {strategy_name} ...")
        for i in range(n_resp):
            sample_rng = make_rng(rng.integers(0, 2**31))
            resp_freq = float(sample_rng.choice(default_respiration_freqs))
            wall_type = wall_type_list[sample_rng.integers(0, len(wall_type_list))]
            scene_id = int(sample_rng.integers(0, n_scenes))
            env_id = int(sample_rng.integers(0, 5))
            snr_db = float(sample_rng.choice(snr_db_choices))
            noise_std = 0.01 + 0.04 * (1.0 - (snr_db + 5) / 35.0)
            noise_std = max(0.005, min(noise_std, 0.1))

            params = {
                'noise_std': noise_std,
                'wall_type': wall_type,
                'respiration_freq': resp_freq,
                'snr_db': snr_db,
                'scene_id': scene_id,
                'environment_id': env_id,
            }

            meta_rng = make_rng(sample_rng.integers(0, 2**31))
            v_meta, _, _ = generate_metasurface_signal(strategy_name, t, rng=meta_rng)

            _, phi, meta = simulate_received_signal(
                t, strategy_name=strategy_name,
                motion_label=0,
                task_type="respiration_estimation",
                rng=sample_rng,
                params=params,
                meta_component=v_meta,
            )

            all_phases.append(phi)
            all_y_motion.append(0)
            all_strategy_idx.append(s_idx)
            all_task_idx.append(1)

            meta_record = {
                'sample_id': sample_id,
                'strategy': strategy_name,
                'motion_label': 0,
                'task_type': 'respiration_estimation',
                'motion_type': 'none',
                'true_motion_freqs': str(meta.get('true_motion_freqs', [])),
                'true_respiration_freq': resp_freq,
                'snr_db': snr_db,
                'human_amplitude': 0.0,
                'metasurface_amplitude': 0.08,
                'noise_std': noise_std,
                'seed': int(sample_rng.integers(0, 2**31)),
                'split': '',
                'scene_id': scene_id,
                'wall_type': wall_type,
                'environment_id': env_id,
            }
            all_metadata.append(meta_record)
            sample_id += 1

    # 转换为数组
    X_phase = np.array(all_phases, dtype=np.float64)
    y_motion = np.array(all_y_motion, dtype=np.int32)
    strategy_index = np.array(all_strategy_idx, dtype=np.int32)
    task_index = np.array(all_task_idx, dtype=np.int32)
    metadata_df = pd.DataFrame(all_metadata)

    # 划分 train/val/test
    n_total = len(y_motion)

    if split_type == "scene_disjoint":
        # Scene-disjoint split: test uses unseen scene_ids
        train_val_scenes = list(range(0, 20))
        test_scenes = list(range(20, n_scenes))

        train_val_mask = metadata_df['scene_id'].isin(train_val_scenes)
        test_mask = metadata_df['scene_id'].isin(test_scenes)

        train_val_indices = np.where(train_val_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Shuffle train_val
        split_rng = make_rng(seed + 1)
        split_rng.shuffle(train_val_indices)

        n_tv = len(train_val_indices)
        n_train = int(n_tv * (train_ratio / (train_ratio + val_ratio)))
        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]

    else:
        # Random split
        indices = np.arange(n_total)
        split_rng = make_rng(seed + 1)
        split_rng.shuffle(indices)

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

    metadata_df.loc[train_indices, 'split'] = 'train'
    metadata_df.loc[val_indices, 'split'] = 'val'
    metadata_df.loc[test_indices, 'split'] = 'test'

    # 保存
    np.savez_compressed(npz_path,
                        X_phase=X_phase,
                        y_motion=y_motion,
                        strategy_index=strategy_index,
                        task_index=task_index)
    metadata_df.to_csv(csv_path, index=False)

    pd.DataFrame({'sample_id': train_indices}).to_csv(
        os.path.join(splits_dir, f'train_ids_{mode}_{split_type}.csv'), index=False)
    pd.DataFrame({'sample_id': val_indices}).to_csv(
        os.path.join(splits_dir, f'val_ids_{mode}_{split_type}.csv'), index=False)
    pd.DataFrame({'sample_id': test_indices}).to_csv(
        os.path.join(splits_dir, f'test_ids_{mode}_{split_type}.csv'), index=False)

    print(f"  Dataset saved: {npz_path}")
    print(f"  Total samples: {n_total} (train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)})")
    if split_type == "scene_disjoint":
        print(f"  Scene-disjoint: train/val scenes={train_val_scenes[:5]}..., test scenes={test_scenes}")

    return {
        'X_phase': X_phase,
        'y_motion': y_motion,
        'strategy_index': strategy_index,
        'task_index': task_index,
        'metadata_df': metadata_df,
    }


def load_dataset(
    mode: str = "debug",
    split_type: str = "random",
    base_dir: str = None,
) -> Dict:
    """加载数据集"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    suffix = f"_{mode}_{split_type}"
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    npz_path = os.path.join(processed_dir, f"rfid_metaprivacy_sim{suffix}.npz")
    csv_path = os.path.join(processed_dir, f"rfid_metaprivacy_metadata{suffix}.csv")

    data = np.load(npz_path)
    metadata_df = pd.read_csv(csv_path)

    return {
        'X_phase': data['X_phase'],
        'y_motion': data['y_motion'],
        'strategy_index': data['strategy_index'],
        'task_index': data['task_index'],
        'metadata_df': metadata_df,
    }


def get_split_indices(split: str, mode: str = "debug", split_type: str = "random",
                      base_dir: str = None) -> np.ndarray:
    """获取某个 split 的 sample_id"""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    splits_dir = os.path.join(base_dir, 'data', 'splits')
    path = os.path.join(splits_dir, f'{split}_ids_{mode}_{split_type}.csv')
    if not os.path.exists(path):
        # fallback to old format
        path = os.path.join(splits_dir, f'{split}_ids.csv')
    df = pd.read_csv(path)
    return df['sample_id'].values


def filter_by_strategy(data: Dict, strategy_name: str,
                       strategies_list: List[str] = None) -> Dict:
    """按策略筛选数据"""
    if strategies_list is None:
        strategies_list = list(strategies)
    s_idx = strategies_list.index(strategy_name)
    mask = data['strategy_index'] == s_idx
    return {
        'X_phase': data['X_phase'][mask],
        'y_motion': data['y_motion'][mask],
        'strategy_index': data['strategy_index'][mask],
        'task_index': data['task_index'][mask],
        'metadata_df': data['metadata_df'][mask].reset_index(drop=True),
    }


def filter_by_task(data: Dict, task_type: str) -> Dict:
    """按任务类型筛选数据"""
    task_map = {'walking_detection': 0, 'respiration_estimation': 1}
    t_idx = task_map.get(task_type, 0)
    mask = data['task_index'] == t_idx
    return {
        'X_phase': data['X_phase'][mask],
        'y_motion': data['y_motion'][mask],
        'strategy_index': data['strategy_index'][mask],
        'task_index': data['task_index'][mask],
        'metadata_df': data['metadata_df'][mask].reset_index(drop=True),
    }


def make_sklearn_arrays(
    data: Dict,
    strategy_name: Optional[str] = None,
    split: Optional[str] = None,
    mode: str = "debug",
    split_type: str = "random",
) -> Tuple[np.ndarray, np.ndarray]:
    """生成 sklearn 格式的特征和标签"""
    X_list = []
    y_list = []

    subset = data
    if strategy_name is not None:
        subset = filter_by_strategy(subset, strategy_name)

    if split is not None:
        split_ids = get_split_indices(split, mode, split_type)
        mask = np.isin(subset['metadata_df']['sample_id'].values, split_ids)
        phases = subset['X_phase'][mask]
        labels = subset['y_motion'][mask]
    else:
        phases = subset['X_phase']
        labels = subset['y_motion']

    for i in range(len(phases)):
        features = extract_features(phases[i])
        X_list.append(features)
        y_list.append(labels[i])

    return np.array(X_list), np.array(y_list)


def make_torch_dataset(data: Dict, strategy_name: Optional[str] = None,
                       split: Optional[str] = None, task_type: str = "walking_detection",
                       mode: str = "debug", split_type: str = "random"):
    """生成 PyTorch Dataset"""
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("  [WARNING] torch not available. Cannot create PyTorch Dataset.")
        return None

    subset = data
    if strategy_name is not None:
        subset = filter_by_strategy(subset, strategy_name)
    if task_type is not None:
        subset = filter_by_task(subset, task_type)

    if split is not None:
        split_ids = get_split_indices(split, mode, split_type)
        mask = np.isin(subset['metadata_df']['sample_id'].values, split_ids)
        phases = subset['X_phase'][mask]
        labels = subset['y_motion'][mask]
    else:
        phases = subset['X_phase']
        labels = subset['y_motion']

    class PhaseDataset(Dataset):
        def __init__(self, phases, labels):
            self.phases = torch.FloatTensor(phases)
            self.labels = torch.LongTensor(labels)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            x = self.phases[idx].unsqueeze(0)
            y = self.labels[idx]
            return x, y

    return PhaseDataset(phases, labels)
