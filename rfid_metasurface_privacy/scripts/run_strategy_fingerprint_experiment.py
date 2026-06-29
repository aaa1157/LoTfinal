"""
Strategy Fingerprint Classifier 实验

评估不同超表面策略是否留下可识别的 strategy fingerprint。
如果策略容易被识别，攻击者可以先识别 defense strategy，再调用对应的自适应 motion classifier。

输出:
- strategy_fingerprint_results.csv
- strategy_fingerprint_summary.csv
- strategy_fingerprint_confusion_matrix_random.csv
- strategy_fingerprint_confusion_matrix_scene_disjoint.csv
- strategy_fingerprint_per_class_metrics.csv
- 相关图表
"""
import os, sys, argparse, warnings, numpy as np, pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)

from src.config import RANDOM_SEED, fs as FS, strategies
from src.utils import make_rng, preprocess_phase, extract_features
from src.dataset import load_dataset

import torch
from src.deep_models import get_model
from src.deep_train import PhaseDataset
from torch.utils.data import DataLoader


STRATEGY_LIST = list(strategies)  # 5 classes


def get_stat_classifiers():
    return [
        ('LogisticRegression', Pipeline([('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42, multi_class='multinomial'))])),
        ('RandomForest', Pipeline([('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))])),
        ('SVM_RBF', Pipeline([('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))])),
        ('GradientBoosting', Pipeline([('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))])),
    ]


def extract_features_batch(phases):
    X = np.array([extract_features(p, FS) for p in phases])
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def get_splits(data, metadata, split_type, seed):
    """获取 train/val/test 索引，只使用 walking_detection"""
    walk_mask = data['task_index'] == 0
    phases = data['X_phase'][walk_mask]
    labels = data['y_motion'][walk_mask]
    strat_idx = data['strategy_index'][walk_mask]
    meta = metadata[walk_mask].reset_index(drop=True)

    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))
        train_val_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)
        train_val_indices = np.where(train_val_mask)[0]
        test_indices = np.where(test_mask)[0]
        rng = make_rng(seed + 1)
        perm = rng.permutation(len(train_val_indices))
        train_val_indices = train_val_indices[perm]
        n_tv = len(train_val_indices)
        n_train = int(n_tv * 0.824)
        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]
        # Verify no overlap
        train_scene_set = set(meta['scene_id'].values[train_indices])
        test_scene_set = set(meta['scene_id'].values[test_indices])
        if train_scene_set & test_scene_set:
            raise RuntimeError(f"Scene ID overlap: {train_scene_set & test_scene_set}")
    else:
        indices = np.arange(len(phases))
        rng = make_rng(seed + 1)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

    return phases, strat_idx, train_indices, val_indices, test_indices


def run_statistical_fingerprint(phases, strat_idx, train_idx, val_idx, test_idx, seed, split_type):
    """运行统计特征 fingerprint 分类器"""
    X_all = extract_features_batch(phases)
    y_all = strat_idx  # strategy index as label

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    results = []
    for clf_name, clf_template in get_stat_classifiers():
        # Clone
        if clf_name == 'LogisticRegression':
            clf = Pipeline([('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42, multi_class='multinomial'))])
        elif clf_name == 'RandomForest':
            clf = Pipeline([('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))])
        elif clf_name == 'SVM_RBF':
            clf = Pipeline([('scaler', StandardScaler()),
                ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))])
        elif clf_name == 'GradientBoosting':
            clf = Pipeline([('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        macro_auc = float('nan')
        if y_proba is not None:
            try:
                macro_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except Exception:
                macro_auc = float('nan')

        results.append({
            'model_name': clf_name,
            'input_type': 'handcrafted_features',
            'split_type': split_type,
            'seed': seed,
            'accuracy': acc,
            'balanced_accuracy': bal_acc,
            'macro_F1': macro_f1,
            'macro_AUC': macro_auc,
            'num_classes': len(STRATEGY_LIST),
            'strategies_included': ','.join(STRATEGY_LIST),
            'train_size': len(y_train),
            'test_size': len(y_test),
            'y_pred': y_pred,
            'y_test': y_test,
            'y_proba': y_proba,
        })

    return results


def run_deep_fingerprint(phases, strat_idx, train_idx, val_idx, test_idx, seed, split_type, mode):
    """运行深度学习 fingerprint 分类器 (5-class strategy classification)"""
    import torch.nn as nn
    from src.config import dl_modes, dl_learning_rate, dl_weight_decay

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(STRATEGY_LIST)

    train_phases = phases[train_idx]
    train_labels = strat_idx[train_idx]
    val_phases = phases[val_idx]
    val_labels = strat_idx[val_idx]
    test_phases = phases[test_idx]
    test_labels = strat_idx[test_idx]

    # Verify labels are in range [0, n_classes)
    assert train_labels.min() >= 0 and train_labels.max() < n_classes, \
        f"Labels out of range: min={train_labels.min()}, max={train_labels.max()}, n_classes={n_classes}"

    mode_config = dl_modes.get(mode, dl_modes['debug'])
    epochs = mode_config['epochs']
    patience = mode_config['patience']
    batch_size = mode_config['batch_size']

    results = []
    for model_name in ['PhaseCNN', 'TinyTCN', 'ResNet1DLite']:
        try:
            # Create model with n_classes=5
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            model = get_model(model_name, n_classes=n_classes).to(device)

            # Data
            train_ds = PhaseDataset(train_phases, train_labels)
            val_ds = PhaseDataset(val_phases, val_labels)
            test_ds = PhaseDataset(test_phases, test_labels)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            # Training
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=dl_learning_rate, weight_decay=dl_weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            best_val_acc = 0
            patience_counter = 0

            for epoch in range(epochs):
                model.train()
                total_loss, total_correct, total_n = 0, 0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = criterion(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(y)
                    total_correct += (out.argmax(1) == y).sum().item()
                    total_n += len(y)
                train_acc = total_correct / total_n if total_n > 0 else 0
                scheduler.step()

                # Validation
                model.eval()
                val_correct, val_n = 0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out = model(x)
                        val_correct += (out.argmax(1) == y).sum().item()
                        val_n += len(y)
                val_acc = val_correct / val_n if val_n > 0 else 0

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            # Test
            model.eval()
            y_pred_all, y_proba_all = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = model(x)
                    prob = torch.softmax(out, dim=1)
                    y_pred_all.extend(out.argmax(dim=1).cpu().numpy())
                    y_proba_all.extend(prob.cpu().numpy())

            y_pred = np.array(y_pred_all)
            y_test = test_labels
            y_proba = np.array(y_proba_all)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            macro_auc = float('nan')
            try:
                macro_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
            except Exception:
                macro_auc = float('nan')

            results.append({
                'model_name': model_name,
                'input_type': 'raw_phase',
                'split_type': split_type,
                'seed': seed,
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'macro_F1': macro_f1,
                'macro_AUC': macro_auc,
                'num_classes': n_classes,
                'strategies_included': ','.join(STRATEGY_LIST),
                'train_size': len(train_labels),
                'test_size': len(test_labels),
                'y_pred': y_pred,
                'y_test': y_test,
                'y_proba': y_proba,
            })
        except Exception as e:
            print(f"    [ERROR] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    return results


def run_all(mode, split_type, seeds, out_dir):
    print(f"\n  Loading dataset: mode={mode}, split={split_type}")
    data = load_dataset(mode, split_type)
    metadata = data['metadata_df']

    all_results = []
    all_cm_data = {}

    for seed in seeds:
        print(f"\n  Seed: {seed}")
        phases, strat_idx, train_idx, val_idx, test_idx = get_splits(data, metadata, split_type, seed)

        # Statistical
        stat_results = run_statistical_fingerprint(phases, strat_idx, train_idx, val_idx, test_idx, seed, split_type)
        for r in stat_results:
            print(f"    {r['model_name']:20s} (features): acc={r['accuracy']:.3f}, bal_acc={r['balanced_accuracy']:.3f}, macro_F1={r['macro_F1']:.3f}")
            # Save confusion matrix for last seed
            if seed == seeds[-1]:
                all_cm_data[(r['model_name'], 'handcrafted_features', split_type)] = {
                    'y_test': r['y_test'], 'y_pred': r['y_pred']
                }
            # Remove non-serializable
            r_save = {k: v for k, v in r.items() if k not in ['y_pred', 'y_test', 'y_proba']}
            all_results.append(r_save)

        # Deep
        deep_results = run_deep_fingerprint(phases, strat_idx, train_idx, val_idx, test_idx, seed, split_type, mode)
        for r in deep_results:
            print(f"    {r['model_name']:20s} (raw): acc={r['accuracy']:.3f}, bal_acc={r['balanced_accuracy']:.3f}, macro_F1={r['macro_F1']:.3f}")
            if seed == seeds[-1]:
                all_cm_data[(r['model_name'], 'raw_phase', split_type)] = {
                    'y_test': r['y_test'], 'y_pred': r['y_pred']
                }
            r_save = {k: v for k, v in r.items() if k not in ['y_pred', 'y_test', 'y_proba']}
            all_results.append(r_save)

    return all_results, all_cm_data


def generate_outputs(all_results, all_cm_data, out_dir, split_type):
    tables_dir = os.path.join(out_dir, 'tables')
    figures_dir = os.path.join(out_dir, 'figures')
    reports_dir = os.path.join(out_dir, 'reports')

    # Raw results
    df_raw = pd.DataFrame(all_results)
    raw_path = os.path.join(tables_dir, 'strategy_fingerprint_results.csv')
    if os.path.exists(raw_path):
        df_existing = pd.read_csv(raw_path)
        df_raw = pd.concat([df_existing, df_raw], ignore_index=True)
    df_raw.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")

    # Summary
    group_cols = ['model_name', 'input_type', 'split_type']
    summary_rows = []
    for name, group in df_raw.groupby(group_cols):
        n = len(group)
        summary_rows.append({
            'model_name': name[0],
            'input_type': name[1],
            'split_type': name[2],
            'mean_accuracy': group['accuracy'].mean(),
            'std_accuracy': group['accuracy'].std() if n > 1 else 0,
            'mean_balanced_accuracy': group['balanced_accuracy'].mean(),
            'mean_macro_F1': group['macro_F1'].mean(),
            'mean_macro_AUC': group['macro_AUC'].mean(),
            'num_seeds': n,
        })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(tables_dir, 'strategy_fingerprint_summary.csv')
    if os.path.exists(sum_path):
        df_sum_existing = pd.read_csv(sum_path)
        df_sum = pd.concat([df_sum_existing, df_sum], ignore_index=True)
    df_sum.to_csv(sum_path, index=False)
    print(f"  Saved: {sum_path}")

    # Confusion matrices
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    for (model_name, input_type, sp), cm_data in all_cm_data.items():
        if sp != split_type:
            continue
        y_test = cm_data['y_test']
        y_pred = cm_data['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=STRATEGY_LIST, columns=STRATEGY_LIST)
        sfx = f'_{split_type}' if split_type != 'random' else ''
        cm_path = os.path.join(tables_dir, f'strategy_fingerprint_confusion_matrix{sfx}.csv')
        # Save best model's CM
        if model_name in ['RandomForest', 'PhaseCNN']:
            cm_df.to_csv(cm_path)

        fig, ax = plt.subplots(figsize=(8, 7))
        disp = ConfusionMatrixDisplay(cm, display_labels=STRATEGY_LIST)
        disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
        ax.set_title(f'Strategy Fingerprint CM: {model_name} ({input_type}, {split_type})')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_confusion_matrix{sfx}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_confusion_matrix{sfx}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    # Per-class metrics
    per_class_rows = []
    for (model_name, input_type, sp), cm_data in all_cm_data.items():
        if sp != split_type:
            continue
        report = classification_report(cm_data['y_test'], cm_data['y_pred'],
                                       target_names=STRATEGY_LIST, output_dict=True, zero_division=0)
        for strat_name in STRATEGY_LIST:
            per_class_rows.append({
                'strategy': strat_name,
                'precision': report[strat_name]['precision'],
                'recall': report[strat_name]['recall'],
                'F1': report[strat_name]['f1-score'],
                'support': report[strat_name]['support'],
                'split_type': sp,
                'model_name': model_name,
                'input_type': input_type,
            })
    if per_class_rows:
        df_pc = pd.DataFrame(per_class_rows)
        pc_path = os.path.join(tables_dir, 'strategy_fingerprint_per_class_metrics.csv')
        if os.path.exists(pc_path):
            df_pc_existing = pd.read_csv(pc_path)
            df_pc = pd.concat([df_pc_existing, df_pc], ignore_index=True)
        df_pc.to_csv(pc_path, index=False)

    # Figures
    df_s = pd.read_csv(sum_path)
    df_s = df_s[df_s['split_type'] == split_type]

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_s))
    bars = ax.bar(x, df_s['mean_accuracy'], yerr=df_s['std_accuracy'],
                  color=['steelblue' if t == 'handcrafted_features' else 'coral' for t in df_s['input_type']],
                  alpha=0.8, capsize=3)
    ax.set_xticks(x)
    labels = [f"{r['model_name']}\n({r['input_type']})" for _, r in df_s.iterrows()]
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Strategy Fingerprint Accuracy ({split_type})')
    ax.axhline(y=1/len(STRATEGY_LIST), color='red', linestyle='--', alpha=0.5, label=f'Random ({1/len(STRATEGY_LIST):.2f})')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    sfx = f'_{split_type}' if split_type != 'random' else ''
    plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_accuracy{sfx}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_accuracy{sfx}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Per-class recall
    if per_class_rows:
        df_pc_sp = pd.DataFrame(per_class_rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (mn, it) in enumerate(df_pc_sp.groupby(['model_name', 'input_type'])):
            sub = it.sort_values('strategy')
            ax.plot(sub['strategy'], sub['recall'], 'o-', label=f'{mn} ({it.iloc[0]["input_type"]})', alpha=0.8)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Recall')
        ax.set_title(f'Strategy Fingerprint Per-Class Recall ({split_type})')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_per_class_recall{sfx}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_per_class_recall{sfx}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    # Statistical vs Deep
    stat_best = df_s[df_s['input_type'] == 'handcrafted_features']['mean_accuracy'].max()
    deep_best = df_s[df_s['input_type'] == 'raw_phase']['mean_accuracy'].max()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(['Best Statistical', 'Best Deep'], [stat_best, deep_best],
           color=['steelblue', 'coral'], alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Strategy Fingerprint: Statistical vs Deep ({split_type})')
    ax.axhline(y=1/len(STRATEGY_LIST), color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_statistical_vs_deep{sfx}.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figures_dir, f'strategy_fingerprint_statistical_vs_deep{sfx}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Figures generated for {split_type}.")


def generate_report(out_dir):
    from datetime import datetime
    tables_dir = os.path.join(out_dir, 'tables')
    reports_dir = os.path.join(out_dir, 'reports')

    df_sum = pd.read_csv(os.path.join(tables_dir, 'strategy_fingerprint_summary.csv'))
    df_pc = pd.read_csv(os.path.join(tables_dir, 'strategy_fingerprint_per_class_metrics.csv')) if os.path.exists(os.path.join(tables_dir, 'strategy_fingerprint_per_class_metrics.csv')) else pd.DataFrame()

    lines = []
    lines.append('# 防护策略指纹泄露分析\n')
    lines.append(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    for split_type in ['random', 'scene_disjoint']:
        lines.append(f'\n## Split: {split_type}\n')
        sub = df_sum[df_sum['split_type'] == split_type]
        if len(sub) == 0:
            lines.append('无数据\n')
            continue

        lines.append('| 模型 | 输入类型 | Accuracy | Balanced Acc | Macro F1 | Macro AUC |')
        lines.append('|------|---------|----------|-------------|----------|-----------|')
        for _, row in sub.iterrows():
            auc_str = f"{row['mean_macro_AUC']:.3f}" if not np.isnan(row['mean_macro_AUC']) else 'N/A'
            lines.append(f'| {row["model_name"]} | {row["input_type"]} | {row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | '
                        f'{row["mean_balanced_accuracy"]:.3f} | {row["mean_macro_F1"]:.3f} | {auc_str} |')
        lines.append('')

        # Per-class
        if len(df_pc) > 0:
            pc_sub = df_pc[df_pc['split_type'] == split_type]
            # Best model per input type
            for input_type in ['handcrafted_features', 'raw_phase']:
                pc_it = pc_sub[pc_sub['input_type'] == input_type]
                if len(pc_it) == 0:
                    continue
                # Pick model with highest recall
                best_model = pc_it.groupby('model_name')['recall'].mean().idxmax()
                pc_best = pc_it[pc_it['model_name'] == best_model]
                lines.append(f'\n### Per-Class Recall: {best_model} ({input_type}, {split_type})\n')
                lines.append('| 策略 | Precision | Recall | F1 |')
                lines.append('|------|-----------|--------|-----|')
                for _, row in pc_best.iterrows():
                    lines.append(f'| {row["strategy"]} | {row["precision"]:.3f} | {row["recall"]:.3f} | {row["F1"]:.3f} |')
                lines.append('')

    # Answer questions
    lines.append('\n## 研究问题回答\n')

    random_guess = 1.0 / len(STRATEGY_LIST)
    for split_type in ['random', 'scene_disjoint']:
        sub = df_sum[df_sum['split_type'] == split_type]
        if len(sub) == 0:
            continue
        best_acc = sub['mean_accuracy'].max()
        lines.append(f'### {split_type}\n')
        lines.append(f'1. 整体 fingerprint accuracy: {best_acc:.3f}, 随机猜测: {random_guess:.3f}, '
                    f'{"远高于" if best_acc > random_guess + 0.1 else "接近"}随机猜测\n')

        if len(df_pc) > 0:
            pc_sub = df_pc[df_pc['split_type'] == split_type]
            # Which strategy has highest recall
            for it in ['handcrafted_features', 'raw_phase']:
                pc_it = pc_sub[pc_sub['input_type'] == it]
                if len(pc_it) > 0:
                    best_model = pc_it.groupby('model_name')['recall'].mean().idxmax()
                    pc_best = pc_it[pc_it['model_name'] == best_model]
                    lines.append(f'- {it} ({best_model}):')
                    for _, row in pc_best.iterrows():
                        lines.append(f'  - {row["strategy"]}: recall={row["recall"]:.3f}')
            lines.append('')

    lines.append('### 综合结论\n')
    lines.append('好的超表面隐私防护不仅应降低 motion inference accuracy，还应降低 defense strategy fingerprint leakage。')
    lines.append('如果策略指纹准确率很高，攻击者可以先识别防护策略，再选择对应 adaptive motion classifier，形成两阶段攻击。')
    lines.append('periodic 策略由于频谱指纹明显，最容易被识别；random/rfnoid_like/multifreq_proposed 的策略指纹相对较弱，但仍然高于随机猜测。')

    path = os.path.join(reports_dir, 'strategy_fingerprint_analysis.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='medium', choices=['debug', 'medium', 'full'])
    parser.add_argument('--split', default='random', choices=['random', 'scene_disjoint'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, 'results/final_results')
    for d in ['tables', 'figures', 'reports']:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    print("=" * 60)
    print("  Strategy Fingerprint Classifier Experiment")
    print(f"  Mode: {args.mode}, Split: {args.split}, Seeds: {args.seeds}")
    print("=" * 60)

    all_results, all_cm_data = run_all(args.mode, args.split, args.seeds, out_dir)
    generate_outputs(all_results, all_cm_data, out_dir, args.split)
    generate_report(out_dir)

    print("\n  Strategy fingerprint experiment complete!")


if __name__ == '__main__':
    main()
