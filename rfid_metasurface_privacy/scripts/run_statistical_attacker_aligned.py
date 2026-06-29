"""
统计攻击者威胁模型对齐实验

与 deep attack 完全相同的威胁模型设置 (Exp A/B/C/D/E)，
使用统计特征分类器 (LR, RF, SVM-RBF, GBM) 进行公平比较。

Usage:
    python scripts/run_statistical_attacker_aligned.py --mode medium --split random --seeds 2026 2027 2028
    python scripts/run_statistical_attacker_aligned.py --mode medium --split scene_disjoint --seeds 2026 2027 2028
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from src.config import (
    RANDOM_SEED, fs as FS, strategies,
)
from src.utils import make_rng, preprocess_phase, extract_features, FEATURE_NAMES
from src.dataset import load_dataset, filter_by_strategy, filter_by_task

# 只用 walking_detection 任务
TASK_TYPE = 'walking_detection'


def get_classifiers():
    """返回分类器列表"""
    return [
        ('LogisticRegression', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
        ])),
        ('RandomForest', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ])),
        ('SVM_RBF', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
        ])),
        ('GradientBoosting', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ])),
    ]


def extract_features_batch(phases):
    """批量提取特征"""
    X = np.array([extract_features(p, FS) for p in phases])
    # 替换 inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X


def compute_detailed_metrics(y_true, y_pred, y_proba=None):
    """计算详细指标"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_acc = (tpr + tnr) / 2

    auc = float('nan')
    if y_proba is not None:
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = float('nan')

    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'TPR': float(tpr),
        'FPR': float(fpr),
        'TNR': float(tnr),
        'FNR': float(fnr),
        'precision': float(precision),
        'recall': float(recall),
        'F1': float(f1),
        'AUC': float(auc),
    }


def get_data_for_experiment(data, metadata, experiment, train_setting, test_strategy,
                            split_type, seed):
    """
    根据实验设置获取 train/test 的 phases 和 labels。
    只使用 walking_detection 任务。

    返回: train_phases, train_labels, test_phases, test_labels,
          train_strategies, test_strategies, train_scenes, test_scenes
    """
    # 只用 walking_detection
    walk_mask = data['task_index'] == 0
    phases = data['X_phase'][walk_mask]
    labels = data['y_motion'][walk_mask]
    strat_idx = data['strategy_index'][walk_mask]
    meta = metadata[walk_mask].reset_index(drop=True)

    strategy_list = list(strategies)

    # 根据 split_type 划分
    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))

        train_val_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)

        train_val_indices = np.where(train_val_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Shuffle train_val
        rng = make_rng(seed + 1)
        perm = rng.permutation(len(train_val_indices))
        train_val_indices = train_val_indices[perm]

        n_tv = len(train_val_indices)
        n_train = int(n_tv * 0.824)  # 70/85 ≈ 0.824
        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]
    else:
        # Random split
        indices = np.arange(len(phases))
        rng = make_rng(seed + 1)
        rng.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

    # 根据 experiment 筛选 train/test
    def filter_by_strategies(indices, strat_names):
        mask = np.zeros(len(indices), dtype=bool)
        for sn in strat_names:
            s_idx = strategy_list.index(sn)
            idx_phases_strat = strat_idx[indices]
            mask |= (idx_phases_strat == s_idx)
        return indices[mask]

    if experiment == 'A':
        # Train: no_metasurface, Test: no_metasurface
        train_sel = filter_by_strategies(train_indices, ['no_metasurface'])
        test_sel = filter_by_strategies(test_indices, ['no_metasurface'])

    elif experiment == 'B':
        # Train: no_metasurface, Test: specific defense
        train_sel = filter_by_strategies(train_indices, ['no_metasurface'])
        test_sel = filter_by_strategies(test_indices, [test_strategy])

    elif experiment == 'C':
        # Train: mixed strategies, Test: each strategy separately
        train_sel = train_indices  # all strategies in train
        test_sel = filter_by_strategies(test_indices, [test_strategy])

    elif experiment == 'D':
        # Train: specific defense, Test: same defense
        train_sel = filter_by_strategies(train_indices, [train_setting])
        test_sel = filter_by_strategies(test_indices, [test_strategy])

    elif experiment == 'E':
        # Leave-one-strategy-out: train on all except target, test on target
        other_strats = [s for s in strategy_list if s != test_strategy]
        train_sel = filter_by_strategies(train_indices, other_strats)
        test_sel = filter_by_strategies(test_indices, [test_strategy])

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    train_phases = phases[train_sel]
    train_labels = labels[train_sel]
    test_phases = phases[test_sel]
    test_labels = labels[test_sel]
    train_strats = strat_idx[train_sel]
    test_strats = strat_idx[test_sel]
    train_scenes = meta['scene_id'].values[train_sel]
    test_scenes = meta['scene_id'].values[test_sel]

    # Scene-disjoint 验证
    if split_type == 'scene_disjoint':
        train_scene_set = set(train_scenes)
        test_scene_set = set(test_scenes)
        overlap = train_scene_set & test_scene_set
        if overlap:
            raise RuntimeError(
                f"Scene ID overlap detected! Overlap: {overlap}. "
                f"Train scenes: {sorted(train_scene_set)[:10]}, "
                f"Test scenes: {sorted(test_scene_set)[:10]}"
            )

    return (train_phases, train_labels, test_phases, test_labels,
            train_strats, test_strats, train_scenes, test_scenes)


def run_all_experiments(mode, split_type, seeds, out_dir):
    """运行所有实验"""
    print(f"\nLoading dataset: mode={mode}, split={split_type}")
    data = load_dataset(mode, split_type)
    metadata = data['metadata_df']

    classifiers = get_classifiers()
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Seed: {seed}")
        print(f"{'='*60}")

        experiments = [
            ('A', 'no_metasurface', 'no_metasurface'),
            ('B', 'no_metasurface', 'periodic'),
            ('B', 'no_metasurface', 'random'),
            ('B', 'no_metasurface', 'rfnoid_like'),
            ('B', 'no_metasurface', 'multifreq_proposed'),
            ('C', 'mixed', 'no_metasurface'),
            ('C', 'mixed', 'periodic'),
            ('C', 'mixed', 'random'),
            ('C', 'mixed', 'rfnoid_like'),
            ('C', 'mixed', 'multifreq_proposed'),
            ('D', 'periodic', 'periodic'),
            ('D', 'random', 'random'),
            ('D', 'rfnoid_like', 'rfnoid_like'),
            ('D', 'multifreq_proposed', 'multifreq_proposed'),
            ('E', 'leave_out', 'no_metasurface'),
            ('E', 'leave_out', 'periodic'),
            ('E', 'leave_out', 'random'),
            ('E', 'leave_out', 'rfnoid_like'),
            ('E', 'leave_out', 'multifreq_proposed'),
        ]

        for exp, train_setting, test_strategy in experiments:
            exp_label = f"Exp {exp}"
            if exp == 'D':
                exp_label = f"Exp D (adaptive_{test_strategy})"
            elif exp == 'E':
                exp_label = f"Exp E (leave_out_{test_strategy})"

            try:
                (train_phases, train_labels, test_phases, test_labels,
                 train_strats, test_strats, train_scenes, test_scenes) = \
                    get_data_for_experiment(
                        data, metadata, exp, train_setting, test_strategy,
                        split_type, seed
                    )
            except RuntimeError as e:
                print(f"  [{exp_label}] ERROR: {e}")
                continue

            if len(train_labels) < 10 or len(test_labels) < 10:
                print(f"  [{exp_label}] SKIP: train={len(train_labels)}, test={len(test_labels)} too small")
                continue

            # 提取特征
            X_train = extract_features_batch(train_phases)
            X_test = extract_features_batch(test_phases)

            for clf_name, clf_pipeline in classifiers:
                try:
                    # 克隆分类器（重新创建以避免状态残留）
                    if clf_name == 'LogisticRegression':
                        clf = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
                        ])
                    elif clf_name == 'RandomForest':
                        clf = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                        ])
                    elif clf_name == 'SVM_RBF':
                        clf = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
                        ])
                    elif clf_name == 'GradientBoosting':
                        clf = Pipeline([
                            ('scaler', StandardScaler()),
                            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
                        ])

                    clf.fit(X_train, train_labels)
                    y_pred = clf.predict(X_test)

                    # 获取概率
                    y_proba = None
                    if hasattr(clf, 'predict_proba'):
                        y_proba = clf.predict_proba(X_test)[:, 1]

                    metrics = compute_detailed_metrics(test_labels, y_pred, y_proba)

                    # Privacy gain: 1 - accuracy (higher = more privacy)
                    privacy_gain = 1.0 - metrics['accuracy']

                    # Train strategy distribution
                    strategy_list = list(strategies)
                    train_strat_counts = {}
                    for s in strategy_list:
                        s_idx = strategy_list.index(s)
                        train_strat_counts[s] = int(np.sum(train_strats == s_idx))

                    test_strat_counts = {}
                    for s in strategy_list:
                        s_idx = strategy_list.index(s)
                        test_strat_counts[s] = int(np.sum(test_strats == s_idx))

                    row = {
                        'classifier_name': clf_name,
                        'experiment': exp,
                        'train_setting': train_setting,
                        'test_strategy': test_strategy,
                        'split_type': split_type,
                        'mode': mode,
                        'seed': seed,
                        'num_train': len(train_labels),
                        'num_test': len(test_labels),
                        'privacy_gain': float(privacy_gain),
                        **metrics,
                        'train_strategy_dist': str(train_strat_counts),
                        'test_strategy_dist': str(test_strat_counts),
                        'train_scene_range': f"{train_scenes.min()}-{train_scenes.max()}" if len(train_scenes) > 0 else "",
                        'test_scene_range': f"{test_scenes.min()}-{test_scenes.max()}" if len(test_scenes) > 0 else "",
                    }
                    all_results.append(row)

                except Exception as e:
                    print(f"  [{exp_label}] {clf_name} ERROR: {e}")
                    continue

            # 只打印一次每个实验的摘要
            best_row = max(
                [r for r in all_results
                 if r['experiment'] == exp and r['train_setting'] == train_setting
                 and r['test_strategy'] == test_strategy and r['seed'] == seed],
                key=lambda r: r['accuracy'],
                default=None
            )
            if best_row:
                print(f"  [{exp_label}] train={len(train_labels)}, test={len(test_labels)}, "
                      f"best={best_row['classifier_name']} acc={best_row['accuracy']:.3f}")

    return all_results


def generate_summary(results, out_dir):
    """生成聚合结果"""
    df = pd.DataFrame(results)

    summary_rows = []
    group_cols = ['classifier_name', 'experiment', 'train_setting', 'test_strategy', 'split_type', 'mode']

    for name, group in df.groupby(group_cols):
        n_seeds = len(group)
        row = {
            'classifier_name': name[0],
            'experiment': name[1],
            'train_setting': name[2],
            'test_strategy': name[3],
            'split_type': name[4],
            'mode': name[5],
            'mean_accuracy': group['accuracy'].mean(),
            'std_accuracy': group['accuracy'].std() if n_seeds > 1 else 0.0,
            'mean_balanced_accuracy': group['balanced_accuracy'].mean(),
            'std_balanced_accuracy': group['balanced_accuracy'].std() if n_seeds > 1 else 0.0,
            'mean_F1': group['F1'].mean(),
            'std_F1': group['F1'].std() if n_seeds > 1 else 0.0,
            'mean_AUC': group['AUC'].mean(),
            'std_AUC': group['AUC'].std() if n_seeds > 1 else 0.0,
            'mean_TPR': group['TPR'].mean(),
            'mean_FPR': group['FPR'].mean(),
            'num_seeds': n_seeds,
        }
        summary_rows.append(row)

    df_sum = pd.DataFrame(summary_rows)
    path = os.path.join(out_dir, 'statistical_attack_aligned_summary.csv')
    df_sum.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df_sum


def generate_fair_comparison(stat_summary_path, deep_summary_path, out_dir, split_type):
    """生成统计 vs 深度公平对比表"""
    df_stat = pd.read_csv(stat_summary_path)
    df_deep = pd.read_csv(deep_summary_path) if os.path.exists(deep_summary_path) else None

    if df_deep is None:
        print("  [WARNING] Deep attack summary not found, skipping fair comparison")
        return None

    # Filter by split_type
    df_stat = df_stat[df_stat['split_type'] == split_type]
    df_deep = df_deep[df_deep['split_type'] == split_type] if 'split_type' in df_deep.columns else df_deep

    comparison_rows = []

    # 对齐 experiment + test_strategy
    for (exp, test_strat), stat_group in df_stat.groupby(['experiment', 'test_strategy']):
        # 找统计分类器最佳
        best_stat = stat_group.loc[stat_group['mean_accuracy'].idxmax()]
        best_stat_acc = best_stat['mean_accuracy']
        best_stat_auc = best_stat['mean_AUC']
        best_stat_clf = best_stat['classifier_name']

        # 找深度模型最佳
        train_setting = best_stat['train_setting']
        deep_match = df_deep[
            (df_deep['train_setting'] == train_setting) &
            (df_deep['test_strategy'] == test_strat)
        ]

        if len(deep_match) > 0:
            best_deep = deep_match.loc[deep_match['mean_accuracy'].idxmax()]
            best_deep_acc = best_deep['mean_accuracy']
            best_deep_auc = best_deep.get('mean_AUC', float('nan'))
            if pd.isna(best_deep_auc):
                best_deep_auc = float('nan')
            best_deep_model = best_deep['model']
        else:
            best_deep_acc = float('nan')
            best_deep_auc = float('nan')
            best_deep_model = 'N/A'

        gap = best_deep_acc - best_stat_acc if not np.isnan(best_deep_acc) else float('nan')

        # Interpretation
        if np.isnan(gap):
            interpretation = "no deep data"
        elif abs(gap) < 0.03:
            interpretation = "comparable"
        elif gap > 0:
            interpretation = "deep stronger"
        else:
            interpretation = "statistical stronger"

        comparison_rows.append({
            'experiment': exp,
            'train_setting': train_setting,
            'test_strategy': test_strat,
            'split_type': split_type,
            'best_statistical_classifier': best_stat_clf,
            'best_statistical_accuracy': round(best_stat_acc, 4),
            'best_deep_model': best_deep_model,
            'best_deep_accuracy': round(best_deep_acc, 4) if not np.isnan(best_deep_acc) else float('nan'),
            'gap_deep_minus_statistical': round(gap, 4) if not np.isnan(gap) else float('nan'),
            'best_statistical_AUC': round(best_stat_auc, 4),
            'best_deep_AUC': round(best_deep_auc, 4) if not np.isnan(best_deep_auc) else float('nan'),
            'interpretation': interpretation,
        })

    df_comp = pd.DataFrame(comparison_rows)
    path = os.path.join(out_dir, 'statistical_vs_deep_fair_comparison.csv')
    df_comp.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df_comp


def generate_figures(out_dir, split_type):
    """生成图表"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tables_dir = os.path.join(out_dir, 'tables')
    figures_dir = os.path.join(out_dir, 'figures')

    df = pd.read_csv(os.path.join(tables_dir, 'statistical_attack_aligned_summary.csv'))
    df = df[df['split_type'] == split_type]

    strategy_order = ['no_metasurface', 'periodic', 'random', 'rfnoid_like', 'multifreq_proposed']
    clf_colors = {
        'LogisticRegression': 'steelblue',
        'RandomForest': 'forestgreen',
        'SVM_RBF': 'coral',
        'GradientBoosting': 'mediumpurple',
    }

    # 1. Statistical attacker threat models overview
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    exp_configs = [
        ('A', 'no_metasurface', 'no_metasurface', 'Exp A: No Defense Baseline'),
        ('B', 'no_metasurface', 'periodic', 'Exp B: Zero-shot (periodic)'),
        ('B', 'no_metasurface', 'random', 'Exp B: Zero-shot (random)'),
        ('C', 'mixed', 'no_metasurface', 'Exp C: Mixed (no_meta)'),
        ('C', 'mixed', 'random', 'Exp C: Mixed (random)'),
        ('C', 'mixed', 'multifreq_proposed', 'Exp C: Mixed (multifreq)'),
    ]

    for idx, (exp, train_s, test_s, title) in enumerate(exp_configs):
        ax = axes[idx // 3, idx % 3]
        subset = df[(df['experiment'] == exp) & (df['train_setting'] == train_s) & (df['test_strategy'] == test_s)]
        if len(subset) == 0:
            ax.set_title(title + '\n(no data)')
            continue

        clfs = subset['classifier_name'].values
        accs = subset['mean_accuracy'].values
        stds = subset['std_accuracy'].values
        colors = [clf_colors.get(c, 'gray') for c in clfs]

        x = np.arange(len(clfs))
        ax.bar(x, accs, yerr=stds, color=colors, alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(clfs, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'statistical_attacker_threat_models.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Statistical vs Deep fair comparison
    comp_path = os.path.join(tables_dir, 'statistical_vs_deep_fair_comparison.csv')
    if os.path.exists(comp_path):
        df_comp = pd.read_csv(comp_path)
        df_comp = df_comp[df_comp['split_type'] == split_type]

        fig, ax = plt.subplots(figsize=(14, 6))
        labels = [f"Exp{r['experiment']}: {r['test_strategy']}" for _, r in df_comp.iterrows()]
        stat_accs = df_comp['best_statistical_accuracy'].values
        deep_accs = df_comp['best_deep_accuracy'].values

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, stat_accs, width, label='Best Statistical', alpha=0.8, color='steelblue')
        deep_accs_plot = np.nan_to_num(deep_accs, nan=0)
        ax.bar(x + width/2, deep_accs_plot, width, label='Best Deep', alpha=0.8, color='coral')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Statistical vs Deep Attacker Fair Comparison ({split_type})')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'statistical_vs_deep_fair_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Zero-shot comparison
    exp_b = df[df['experiment'] == 'B']
    if len(exp_b) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for clf_name in exp_b['classifier_name'].unique():
            sub = exp_b[exp_b['classifier_name'] == clf_name]
            strats = sub['test_strategy'].values
            accs = sub['mean_accuracy'].values
            ax.plot(strats, accs, 'o-', label=clf_name, color=clf_colors.get(clf_name, 'gray'), alpha=0.8)

        ax.set_xlabel('Test Strategy')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Zero-Shot Statistical Attacker (train=no_meta, {split_type})')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'zero_shot_statistical_vs_deep.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Seen-defense comparison
    exp_d = df[df['experiment'] == 'D']
    if len(exp_d) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for clf_name in exp_d['classifier_name'].unique():
            sub = exp_d[exp_d['classifier_name'] == clf_name]
            strats = sub['test_strategy'].values
            accs = sub['mean_accuracy'].values
            ax.bar(np.arange(len(strats)) - 0.1 * list(exp_d['classifier_name'].unique()).index(clf_name),
                   accs, width=0.2, label=clf_name, color=clf_colors.get(clf_name, 'gray'), alpha=0.8)

        ax.set_xlabel('Defense Strategy')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Seen-Defense Statistical Attacker (Exp D, {split_type})')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'seen_defense_statistical_vs_deep.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Mixed-defense comparison
    exp_c = df[df['experiment'] == 'C']
    if len(exp_c) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for clf_name in exp_c['classifier_name'].unique():
            sub = exp_c[exp_c['classifier_name'] == clf_name]
            strats = sub['test_strategy'].values
            accs = sub['mean_accuracy'].values
            ax.plot(strats, accs, 'o-', label=clf_name, color=clf_colors.get(clf_name, 'gray'), alpha=0.8)

        ax.set_xlabel('Test Strategy')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Mixed-Defense Statistical Attacker (Exp C, {split_type})')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'mixed_defense_statistical_vs_deep.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print("  Figures generated.")


def generate_report(out_dir, split_type):
    """生成分析报告"""
    from datetime import datetime

    tables_dir = os.path.join(out_dir, 'tables')
    reports_dir = os.path.join(out_dir, 'reports')

    df_sum = pd.read_csv(os.path.join(tables_dir, 'statistical_attack_aligned_summary.csv'))
    df_sum = df_sum[df_sum['split_type'] == split_type]

    comp_path = os.path.join(tables_dir, 'statistical_vs_deep_fair_comparison.csv')
    df_comp = pd.read_csv(comp_path) if os.path.exists(comp_path) else pd.DataFrame()

    lines = []
    lines.append('# 统计攻击者 vs 深度攻击者公平对比分析\n')
    lines.append(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    lines.append(f'> 数据集: medium, split={split_type}, seeds=2026/2027/2028\n')
    lines.append(f'> 分类器: LogisticRegression, RandomForest, SVM-RBF, GradientBoosting\n')

    # Exp A
    lines.append('## 1. Experiment A: 无防护基线\n')
    exp_a = df_sum[(df_sum['experiment'] == 'A') & (df_sum['test_strategy'] == 'no_metasurface')]
    if len(exp_a) > 0:
        lines.append('| 分类器 | Accuracy | Balanced Acc | AUC | F1 |')
        lines.append('|--------|----------|-------------|-----|-----|')
        for _, row in exp_a.iterrows():
            lines.append(f'| {row["classifier_name"]} | {row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | '
                        f'{row["mean_balanced_accuracy"]:.3f} | {row["mean_AUC"]:.3f} | {row["mean_F1"]:.3f} |')
    lines.append('')

    # Exp B
    lines.append('## 2. Experiment B: 零样本跨策略\n')
    exp_b = df_sum[df_sum['experiment'] == 'B']
    if len(exp_b) > 0:
        lines.append('| 分类器 | 测试策略 | Accuracy | AUC |')
        lines.append('|--------|---------|----------|-----|')
        for _, row in exp_b.iterrows():
            lines.append(f'| {row["classifier_name"]} | {row["test_strategy"]} | '
                        f'{row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | {row["mean_AUC"]:.3f} |')
    lines.append('')

    # Exp C
    lines.append('## 3. Experiment C: 混合策略训练\n')
    exp_c = df_sum[df_sum['experiment'] == 'C']
    if len(exp_c) > 0:
        lines.append('| 分类器 | 测试策略 | Accuracy | AUC |')
        lines.append('|--------|---------|----------|-----|')
        for _, row in exp_c.iterrows():
            lines.append(f'| {row["classifier_name"]} | {row["test_strategy"]} | '
                        f'{row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | {row["mean_AUC"]:.3f} |')
    lines.append('')

    # Exp D
    lines.append('## 4. Experiment D: 自适应/已见防护\n')
    exp_d = df_sum[df_sum['experiment'] == 'D']
    if len(exp_d) > 0:
        lines.append('| 分类器 | 防护策略 | Accuracy | AUC |')
        lines.append('|--------|---------|----------|-----|')
        for _, row in exp_d.iterrows():
            lines.append(f'| {row["classifier_name"]} | {row["test_strategy"]} | '
                        f'{row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | {row["mean_AUC"]:.3f} |')
    lines.append('')

    # Exp E
    lines.append('## 5. Experiment E: Leave-One-Out\n')
    exp_e = df_sum[df_sum['experiment'] == 'E']
    if len(exp_e) > 0:
        lines.append('| 分类器 | 留出策略 | Accuracy | AUC |')
        lines.append('|--------|---------|----------|-----|')
        for _, row in exp_e.iterrows():
            lines.append(f'| {row["classifier_name"]} | {row["test_strategy"]} | '
                        f'{row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | {row["mean_AUC"]:.3f} |')
    lines.append('')

    # Fair comparison
    lines.append('## 6. 统计 vs 深度公平对比\n')
    if len(df_comp) > 0:
        lines.append('| 实验 | 测试策略 | 最佳统计分类器 | 统计准确率 | 最佳深度模型 | 深度准确率 | 差距 | 解读 |')
        lines.append('|------|---------|-------------|----------|-----------|----------|------|------|')
        for _, row in df_comp.iterrows():
            lines.append(f'| {row["experiment"]} | {row["test_strategy"]} | {row["best_statistical_classifier"]} | '
                        f'{row["best_statistical_accuracy"]:.3f} | {row["best_deep_model"]} | '
                        f'{row["best_deep_accuracy"]:.3f} | {row["gap_deep_minus_statistical"]:+.3f} | '
                        f'{row["interpretation"]} |')
    lines.append('')

    # Answer questions
    lines.append('## 7. 研究问题回答\n')

    # Q1
    lines.append('### Q1: 零样本设置下，深度模型是否优于统计分类器？\n')
    if len(df_comp) > 0:
        exp_b_comp = df_comp[df_comp['experiment'] == 'B']
        if len(exp_b_comp) > 0:
            for _, row in exp_b_comp.iterrows():
                gap = row['gap_deep_minus_statistical']
                if np.isnan(gap):
                    lines.append(f'- {row["test_strategy"]}: 无深度数据')
                elif gap > 0.05:
                    lines.append(f'- {row["test_strategy"]}: 深度模型显著更强 (gap={gap:+.3f})')
                elif gap > 0:
                    lines.append(f'- {row["test_strategy"]}: 深度模型略强 (gap={gap:+.3f})')
                else:
                    lines.append(f'- {row["test_strategy"]}: 统计分类器更强或相当 (gap={gap:+.3f})')
    lines.append('')

    # Q2
    lines.append('### Q2: 已见防护设置下，统计分类器是否明显变强？\n')
    if len(exp_d) > 0 and len(exp_b) > 0:
        for strat in ['periodic', 'random', 'rfnoid_like', 'multifreq_proposed']:
            b_best = exp_b[exp_b['test_strategy'] == strat]['mean_accuracy'].max()
            d_best = exp_d[exp_d['test_strategy'] == strat]['mean_accuracy'].max()
            if not np.isnan(b_best) and not np.isnan(d_best):
                gain = d_best - b_best
                lines.append(f'- {strat}: zero-shot最佳={b_best:.3f}, seen-defense最佳={d_best:.3f}, '
                            f'增益={gain:+.3f}')
    lines.append('')

    # Q3
    lines.append('### Q3: 混合策略训练下，统计分类器是否也能适应？\n')
    if len(exp_c) > 0:
        for strat in list(strategies):
            c_best = exp_c[exp_c['test_strategy'] == strat]['mean_accuracy'].max()
            if not np.isnan(c_best):
                lines.append(f'- {strat}: mixed最佳={c_best:.3f}')
    lines.append('')

    # Q4
    lines.append('### Q4: 深度模型高准确率来自"模型能力"还是"训练分布"？\n')
    lines.append('- 需要对比 Exp B (zero-shot) 和 Exp C (mixed) 的差距')
    lines.append('- 如果 Exp C >> Exp B，说明训练分布是关键因素')
    lines.append('- 如果 Exp C ≈ Exp B，说明模型能力是关键因素')
    lines.append('')

    # Q5-Q7
    lines.append('### Q5-Q7: 综合结论\n')
    if len(df_comp) > 0:
        deep_stronger = (df_comp['gap_deep_minus_statistical'] > 0.05).sum()
        stat_stronger = (df_comp['gap_deep_minus_statistical'] < -0.05).sum()
        comparable = len(df_comp) - deep_stronger - stat_stronger
        lines.append(f'- 深度模型显著更强: {deep_stronger} 个场景')
        lines.append(f'- 统计分类器显著更强: {stat_stronger} 个场景')
        lines.append(f'- 相当: {comparable} 个场景')
    lines.append('')

    # Limitations
    lines.append('## 8. 局限性\n')
    lines.append('1. 统计特征为手工设计，可能遗漏重要模式')
    lines.append('2. 分类器超参数未做精细调优')
    lines.append('3. 机理级仿真，非真实数据')
    lines.append('4. medium 数据集规模有限')
    lines.append('')

    path = os.path.join(reports_dir, 'statistical_vs_deep_fair_analysis.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Statistical Attacker Aligned Experiments")
    parser.add_argument('--mode', type=str, default='medium', choices=['debug', 'medium', 'full'])
    parser.add_argument('--split', type=str, default='random', choices=['random', 'scene_disjoint'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, 'results/final_results')
    tables_dir = os.path.join(out_dir, 'tables')
    figures_dir = os.path.join(out_dir, 'figures')
    reports_dir = os.path.join(out_dir, 'reports')

    for d in [tables_dir, figures_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  Statistical Attacker Aligned Experiments")
    print(f"  Mode: {args.mode}, Split: {args.split}, Seeds: {args.seeds}")
    print("=" * 60)

    # Run experiments
    results = run_all_experiments(args.mode, args.split, args.seeds, tables_dir)

    # Save raw results
    df_raw = pd.DataFrame(results)
    raw_path = os.path.join(tables_dir, 'statistical_attack_aligned_results.csv')
    df_raw.to_csv(raw_path, index=False)
    print(f"\nSaved: {raw_path}")

    # Generate summary
    df_sum = generate_summary(results, tables_dir)

    # Generate fair comparison
    deep_summary_path = os.path.join(tables_dir, f'deep_attack_summary_{args.mode}_{args.split}.csv')
    generate_fair_comparison(
        os.path.join(tables_dir, 'statistical_attack_aligned_summary.csv'),
        deep_summary_path,
        tables_dir,
        args.split,
    )

    # Generate figures
    generate_figures(out_dir, args.split)

    # Generate report
    generate_report(out_dir, args.split)

    print("\n  Statistical attacker aligned experiments complete!")


if __name__ == '__main__':
    main()
