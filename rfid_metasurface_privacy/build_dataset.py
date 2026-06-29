"""
RFID-MetaPrivacy-Sim 数据集构建脚本

Usage:
    python build_dataset.py --mode debug
    python build_dataset.py --mode medium --split scene_disjoint
    python build_dataset.py --mode full --split scene_disjoint --force
    python build_dataset.py --fast  (alias for --mode debug)
"""

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import RANDOM_SEED
from src.dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Build RFID-MetaPrivacy-Sim Dataset")
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['debug', 'medium', 'full'],
                        help='Dataset mode: debug/medium/full')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'scene_disjoint'],
                        help='Split type: random/scene_disjoint')
    parser.add_argument('--fast', action='store_true', help='Alias for --mode debug')
    parser.add_argument('--force', action='store_true', help='Force rebuild')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')
    args = parser.parse_args()

    mode = 'debug' if args.fast else args.mode

    print("=" * 60)
    print("  RFID-MetaPrivacy-Sim Dataset Builder")
    print("=" * 60)
    print(f"  Mode: {mode}")
    print(f"  Split: {args.split}")
    print(f"  Seed: {args.seed}")
    print()

    data = build_dataset(mode=mode, split_type=args.split, force=args.force,
                         seed=args.seed, base_dir=PROJECT_ROOT)

    print()
    print("  Dataset built successfully!")
    print(f"  Total samples: {len(data['y_motion'])}")
    print(f"  Walking: {int(np.sum(data['task_index'] == 0))}")
    print(f"  Respiration: {int(np.sum(data['task_index'] == 1))}")


if __name__ == '__main__':
    main()
