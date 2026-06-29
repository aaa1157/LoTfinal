"""
Phase 2: 深度学习攻击者训练脚本

实验 A: no defense baseline
实验 B: cross-strategy generalization
实验 C: mixed-strategy attacker
实验 D: adaptive attacker per defense
实验 E: leave-one-strategy-out

Usage:
    python train_deep_attacker.py --mode debug
    python train_deep_attacker.py --mode medium --seeds 2026 2027 2028
    python train_deep_attacker.py --mode full --seeds 2026 2027 2028
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
    print("[ERROR] PyTorch is not available. Deep learning experiments cannot run.")
    print("  Install with: pip install torch")
    sys.exit(0)

from src.config import strategies, RANDOM_SEED
from src.dataset import load_dataset, filter_by_strategy, filter_by_task, get_split_indices
from src.deep_train import train_model, test_model
from src.utils import make_rng

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


ALL_MODELS = ['PhaseCNN', 'PhaseNetLite', 'TinyTCN', 'ResNet1DLite', 'DualBranchNet']
DEBUG_MODELS = ['PhaseCNN', 'PhaseNetLite']


def run_single_experiment(
    model_name: str,
    train_phases, train_labels,
    val_phases, val_labels,
    test_phases, test_labels,
    train_setting: str,
    test_strategy: str,
    mode: str = "debug",
    seed: int = 2026,
) -> dict:
    """运行单个实验"""
    result = train_model(
        model_name, train_phases, train_labels,
        val_phases, val_labels,
        mode=mode,
        save_dir=os.path.join(PROJECT_ROOT, 'results/models'),
        log_dir=os.path.join(PROJECT_ROOT, 'results/logs'),
        seed=seed,
    )

    test_metrics = test_model(result['model'], test_phases, test_labels)

    return {
        'model': model_name,
        'train_setting': train_setting,
        'test_strategy': test_strategy,
        'seed': seed,
        'accuracy': test_metrics['accuracy'],
        'TPR': test_metrics['tpr'],
        'FPR': test_metrics['fpr'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'f1': test_metrics['f1'],
        'params': result['n_params'],
        'inference_time_ms': result['inference_time_ms'],
        'device': result['device'],
        'epochs_trained': result['epochs_trained'],
        'train_history': result['train_history'],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Deep Learning Attacker")
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['debug', 'medium', 'full'])
    parser.add_argument('--fast', action='store_true', help='Alias for --mode debug')
    parser.add_argument('--model', type=str, default='all',
                        help='Model name or all')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026],
                        help='Random seeds for multi-seed experiments')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'scene_disjoint'])
    args = parser.parse_args()

    mode = 'debug' if args.fast else args.mode

    print("=" * 60)
    print("  Phase 2: Deep Learning Attacker")
    print("=" * 60)
    print(f"  Mode: {mode}, Split: {args.split}, Seeds: {args.seeds}")

    # Load dataset
    print("\n[1] Loading dataset ...")
    data = load_dataset(mode=mode, split_type=args.split, base_dir=PROJECT_ROOT)

    # Model list
    if args.model == 'all':
        model_names = DEBUG_MODELS if mode == 'debug' else ALL_MODELS
    else:
        model_names = [args.model]

    all_results = []

    for seed in args.seeds:
        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"  Model: {model_name}, Seed: {seed}")
            print(f"{'='*50}")

            # Get split indices
            train_ids = get_split_indices('train', mode, args.split, PROJECT_ROOT)
            val_ids = get_split_indices('val', mode, args.split, PROJECT_ROOT)
            test_ids = get_split_indices('test', mode, args.split, PROJECT_ROOT)

            # ---- Experiment A: Same strategy ----
            print("\n  [Exp A] Train: no_metasurface, Test: no_metasurface")
            no_meta = filter_by_strategy(
                filter_by_task(data, 'walking_detection'), 'no_metasurface'
            )
            meta_df = no_meta['metadata_df']
            train_mask = meta_df['sample_id'].isin(train_ids)
            val_mask = meta_df['sample_id'].isin(val_ids)
            test_mask = meta_df['sample_id'].isin(test_ids)

            result_a = run_single_experiment(
                model_name,
                no_meta['X_phase'][train_mask], no_meta['y_motion'][train_mask],
                no_meta['X_phase'][val_mask], no_meta['y_motion'][val_mask],
                no_meta['X_phase'][test_mask], no_meta['y_motion'][test_mask],
                train_setting='no_metasurface',
                test_strategy='no_metasurface',
                mode=mode, seed=seed,
            )
            all_results.append(result_a)
            print(f"    Acc={result_a['accuracy']:.3f}, TPR={result_a['TPR']:.3f}, FPR={result_a['FPR']:.3f}")

            # ---- Experiment B: Cross-strategy ----
            print("\n  [Exp B] Train: no_metasurface, Test: other strategies")
            for test_strat in strategies[1:]:
                strat_data = filter_by_strategy(
                    filter_by_task(data, 'walking_detection'), test_strat
                )
                strat_meta = strat_data['metadata_df']
                strat_test_mask = strat_meta['sample_id'].isin(test_ids)

                result_b = run_single_experiment(
                    model_name,
                    no_meta['X_phase'][train_mask], no_meta['y_motion'][train_mask],
                    no_meta['X_phase'][val_mask], no_meta['y_motion'][val_mask],
                    strat_data['X_phase'][strat_test_mask], strat_data['y_motion'][strat_test_mask],
                    train_setting='no_metasurface',
                    test_strategy=test_strat,
                    mode=mode, seed=seed,
                )
                all_results.append(result_b)
                print(f"    Test={test_strat}: Acc={result_b['accuracy']:.3f}")

            # ---- Experiment C: Mixed strategy ----
            print("\n  [Exp C] Train: mixed strategies, Test: all strategies")
            walk_data = filter_by_task(data, 'walking_detection')
            walk_meta = walk_data['metadata_df']
            walk_train_mask = walk_meta['sample_id'].isin(train_ids)
            walk_val_mask = walk_meta['sample_id'].isin(val_ids)

            for test_strat in strategies:
                strat_data = filter_by_strategy(
                    filter_by_task(data, 'walking_detection'), test_strat
                )
                strat_meta = strat_data['metadata_df']
                strat_test_mask = strat_meta['sample_id'].isin(test_ids)

                result_c = run_single_experiment(
                    model_name,
                    walk_data['X_phase'][walk_train_mask], walk_data['y_motion'][walk_train_mask],
                    walk_data['X_phase'][walk_val_mask], walk_data['y_motion'][walk_val_mask],
                    strat_data['X_phase'][strat_test_mask], strat_data['y_motion'][strat_test_mask],
                    train_setting='mixed',
                    test_strategy=test_strat,
                    mode=mode, seed=seed,
                )
                all_results.append(result_c)
                print(f"    Test={test_strat}: Acc={result_c['accuracy']:.3f}")

            # ---- Experiment D: Adaptive attacker per defense ----
            print("\n  [Exp D] Adaptive attacker per defense")
            for train_strat in strategies[1:]:
                strat_train_data = filter_by_strategy(
                    filter_by_task(data, 'walking_detection'), train_strat
                )
                strat_train_meta = strat_train_data['metadata_df']
                strat_train_mask = strat_train_meta['sample_id'].isin(train_ids)
                strat_val_mask = strat_train_meta['sample_id'].isin(val_ids)
                strat_test_mask = strat_train_meta['sample_id'].isin(test_ids)

                result_d = run_single_experiment(
                    model_name,
                    strat_train_data['X_phase'][strat_train_mask], strat_train_data['y_motion'][strat_train_mask],
                    strat_train_data['X_phase'][strat_val_mask], strat_train_data['y_motion'][strat_val_mask],
                    strat_train_data['X_phase'][strat_test_mask], strat_train_data['y_motion'][strat_test_mask],
                    train_setting=f'adaptive_{train_strat}',
                    test_strategy=train_strat,
                    mode=mode, seed=seed,
                )
                all_results.append(result_d)
                print(f"    Train=Test={train_strat}: Acc={result_d['accuracy']:.3f}")

            # ---- Experiment E: Leave-one-strategy-out ----
            print("\n  [Exp E] Leave-one-strategy-out")
            for held_out in strategies:
                # Train on all except held_out
                other_strats = [s for s in strategies if s != held_out]
                other_data_list = []
                other_labels_list = []
                other_val_list = []
                other_val_labels = []

                for s in other_strats:
                    sd = filter_by_strategy(filter_by_task(data, 'walking_detection'), s)
                    sm = sd['metadata_df']
                    s_train = sm['sample_id'].isin(train_ids)
                    s_val = sm['sample_id'].isin(val_ids)
                    other_data_list.append(sd['X_phase'][s_train])
                    other_labels_list.append(sd['y_motion'][s_train])
                    other_val_list.append(sd['X_phase'][s_val])
                    other_val_labels.append(sd['y_motion'][s_val])

                other_train_phases = np.concatenate(other_data_list)
                other_train_labels = np.concatenate(other_labels_list)
                other_val_phases = np.concatenate(other_val_list)
                other_val_labels = np.concatenate(other_val_labels)

                held_data = filter_by_strategy(
                    filter_by_task(data, 'walking_detection'), held_out
                )
                held_meta = held_data['metadata_df']
                held_test_mask = held_meta['sample_id'].isin(test_ids)

                result_e = run_single_experiment(
                    model_name,
                    other_train_phases, other_train_labels,
                    other_val_phases, other_val_labels,
                    held_data['X_phase'][held_test_mask], held_data['y_motion'][held_test_mask],
                    train_setting=f'leave_out_{held_out}',
                    test_strategy=held_out,
                    mode=mode, seed=seed,
                )
                all_results.append(result_e)
                print(f"    Leave-out={held_out}: Acc={result_e['accuracy']:.3f}")

    # Save results
    print("\n[2] Saving results ...")
    results_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != 'train_history'}
        results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/deep_attack_results.csv'), index=False)
    print("  Saved: results/tables/deep_attack_results.csv")

    # Summary
    summary_rows = []
    for model_name in model_names:
        for train_setting in results_df['train_setting'].unique():
            for test_strategy in results_df['test_strategy'].unique():
                subset = results_df[
                    (results_df['model'] == model_name) &
                    (results_df['train_setting'] == train_setting) &
                    (results_df['test_strategy'] == test_strategy)
                ]
                if len(subset) > 0:
                    summary_rows.append({
                        'model': model_name,
                        'train_setting': train_setting,
                        'test_strategy': test_strategy,
                        'mean_accuracy': subset['accuracy'].mean(),
                        'std_accuracy': subset['accuracy'].std(),
                        'mean_f1': subset['f1'].mean(),
                        'std_f1': subset['f1'].std(),
                        'mean_tpr': subset['TPR'].mean(),
                        'mean_fpr': subset['FPR'].mean(),
                        'n_seeds': len(subset),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/deep_attack_summary.csv'), index=False)
    print("  Saved: results/tables/deep_attack_summary.csv")

    # Plot
    print("\n[3] Plotting ...")

    # Deep attacker comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    exp_b = results_df[results_df['train_setting'] == 'no_metasurface']
    if len(exp_b) > 0:
        for model_name in model_names:
            model_b = exp_b[exp_b['model'] == model_name]
            if len(model_b) > 0:
                axes[0].bar(model_b['test_strategy'] + f' ({model_name})',
                           model_b['accuracy'], alpha=0.7, label=model_name)
        axes[0].set_title('Exp B: Cross-Strategy Generalization')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1.1)
        axes[0].legend(fontsize=7)
        axes[0].tick_params(axis='x', rotation=45, labelsize=7)
        axes[0].grid(True, alpha=0.3, axis='y')

    exp_c = results_df[results_df['train_setting'] == 'mixed']
    if len(exp_c) > 0:
        for model_name in model_names:
            model_c = exp_c[exp_c['model'] == model_name]
            if len(model_c) > 0:
                axes[1].bar(model_c['test_strategy'] + f' ({model_name})',
                           model_c['accuracy'], alpha=0.7, label=model_name)
        axes[1].set_title('Exp C: Mixed Strategy Training')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim(0, 1.1)
        axes[1].legend(fontsize=7)
        axes[1].tick_params(axis='x', rotation=45, labelsize=7)
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/deep_attacker_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/deep_attacker_comparison.png")

    # Training curves
    fig, axes = plt.subplots(len(model_names), 1, figsize=(10, 4 * len(model_names)))
    if len(model_names) == 1:
        axes = [axes]

    for idx, model_name in enumerate(model_names):
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            history = model_results[0]['train_history']
            epochs = [h['epoch'] for h in history]
            train_acc = [h['train_acc'] for h in history]
            val_acc = [h['val_acc'] for h in history]
            axes[idx].plot(epochs, train_acc, label='Train Acc', color='steelblue')
            axes[idx].plot(epochs, val_acc, label='Val Acc', color='orangered')
            axes[idx].set_title(f'{model_name} Training Curves')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Accuracy')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results/figures/deep_attacker_train_curves.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: results/figures/deep_attacker_train_curves.png")

    # Print summary
    print("\n" + "=" * 60)
    print("  Phase 2 Results Summary")
    print("=" * 60)
    if len(summary_df) > 0:
        print(f"\n{'Model':<15} {'Train':<20} {'Test':<22} {'Acc':>6} {'F1':>6} {'Seeds':>5}")
        print("-" * 75)
        for _, row in summary_df.iterrows():
            print(f"{row['model']:<15} {row['train_setting']:<20} {row['test_strategy']:<22} "
                  f"{row['mean_accuracy']:>6.3f} {row['mean_f1']:>6.3f} {row['n_seeds']:>5d}")

    print("\n  Phase 2 complete!")


if __name__ == '__main__':
    main()
