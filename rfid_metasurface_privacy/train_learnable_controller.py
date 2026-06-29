"""
Phase 3: 可学习超表面控制器训练

Usage:
    python train_learnable_controller.py --mode debug
    python train_learnable_controller.py --mode medium
    python train_learnable_controller.py --mode full
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch is not available.")
    sys.exit(0)

from src.config import strategies, RANDOM_SEED
from src.deep_models import get_model
from src.learnable_controller import LMCController

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Learnable Metasurface Controller")
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['debug', 'medium', 'full'])
    parser.add_argument('--fast', action='store_true', help='Alias for --mode debug')
    parser.add_argument('--attacker-model', type=str, default='PhaseCNN',
                        help='Frozen attacker model name')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'scene_disjoint'])
    args = parser.parse_args()

    mode = 'debug' if args.fast else args.mode

    print("=" * 60)
    print("  Phase 3: Learnable Metasurface Controller")
    print("=" * 60)
    print(f"  Mode: {mode}, Attacker: {args.attacker_model}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load frozen attacker
    print("\n[1] Loading frozen attacker model ...")
    attacker = get_model(args.attacker_model).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'results/models', f'{args.attacker_model}_best.pt')
    if os.path.exists(model_path):
        attacker.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded: {model_path}")
    else:
        print(f"  [WARNING] No saved model at {model_path}, using random init")
    attacker.eval()

    # Run LMC search
    print("\n[2] Running LMC search ...")
    controller = LMCController(seed=RANDOM_SEED)
    result = controller.search(attacker, device, mode=mode)

    best_params = result['best_params']
    best_metrics = result['best_metrics']
    search_history = result['search_history']

    # Save search history
    print("\n[3] Saving results ...")
    history_df = pd.DataFrame(search_history)
    history_df.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/controller_search_history.csv'), index=False)
    print("  Saved: results/tables/controller_search_history.csv")

    # Save best params
    params_df = pd.DataFrame([{
        'param': k,
        'value': str(v) if isinstance(v, (list, tuple)) else v,
    } for k, v in best_params.items()])
    params_df.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/lmc_best_params.csv'), index=False)
    print("  Saved: results/tables/lmc_best_params.csv")

    # Save metrics
    metrics_row = {
        'controller': 'LMC',
        'attacker_model': args.attacker_model,
        **best_metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(
        os.path.join(PROJECT_ROOT, 'results/tables/lmc_metrics.csv'), index=False)
    print("  Saved: results/tables/lmc_metrics.csv")

    # Plot search history
    print("\n[4] Plotting ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if len(search_history) > 0:
        iters = [h['iter'] for h in search_history]
        objs = [h['objective'] for h in search_history]
        accs = [h['attacker_accuracy'] for h in search_history]
        ents = [h.get('phase_entropy', 0) for h in search_history]
        lfs = [h.get('lowfreq_energy_ratio', 0) for h in search_history]

        axes[0, 0].plot(iters, objs, 'b-', alpha=0.5)
        axes[0, 0].set_title('Objective over Search')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Objective')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(iters, accs, 'r-', alpha=0.5)
        axes[0, 1].set_title('Attacker Accuracy over Search')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(iters, ents, 'g-', alpha=0.5)
        axes[1, 0].set_title('Phase Entropy over Search')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Entropy')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(iters, lfs, 'm-', alpha=0.5)
        axes[1, 1].set_title('Low-freq Energy Ratio over Search')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('LF Ratio')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/lmc_search_history.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/lmc_search_history.png")

    # Print summary
    print("\n" + "=" * 60)
    print("  Phase 3 Results Summary")
    print("=" * 60)
    print(f"\n  Best Objective: {best_metrics['objective']:.4f}")
    print(f"  Best Attacker Accuracy: {best_metrics['attacker_accuracy']:.3f}")
    print(f"  Best Phase Entropy: {best_metrics.get('phase_entropy', 0):.3f}")
    print(f"  Best LF Ratio: {best_metrics.get('lowfreq_energy_ratio', 0):.3f}")
    print(f"  Best Switching Rate: {best_metrics.get('switching_rate', 0):.3f}")
    print(f"\n  Best Params:")
    for k, v in best_params.items():
        print(f"    {k}: {v}")

    print("\n  Phase 3 complete!")


if __name__ == '__main__':
    main()
