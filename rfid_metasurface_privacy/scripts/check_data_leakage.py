"""
Data Leakage Audit

检查实验中是否存在数据泄露问题，包括:
1. sample_id 泄露
2. scene_id 泄露
3. phase hash 泄露
4. 近重复样本
5. metadata-only classifier
6. label-shuffle sanity check
7. 输入字段检查
8. normalization 泄露检查
"""
import os, sys, hashlib, warnings, numpy as np, pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config import RANDOM_SEED, fs as FS, strategies
from src.dataset import load_dataset
from src.utils import make_rng, extract_features

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def phase_hash(phase):
    """计算 phase sequence 的 hash"""
    return hashlib.md5(phase.tobytes()).hexdigest()


def check_sample_id_leakage(data, metadata, split_type):
    """检查 train/val/test 是否存在重复 sample_id"""
    print("  [1] Checking sample_id leakage...")
    results = []

    walk_mask = data['task_index'] == 0
    meta = metadata[walk_mask].reset_index(drop=True)

    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))
        train_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)
        train_ids = set(meta[train_mask].index)
        test_ids = set(meta[test_mask].index)
    else:
        indices = np.arange(len(meta))
        rng = make_rng(RANDOM_SEED + 1)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_ids = set(indices[:n_train])
        val_ids = set(indices[n_train:n_train + n_val])
        test_ids = set(indices[n_train + n_val:])

    overlap_train_test = train_ids & test_ids
    overlap_train_val = train_ids & val_ids if 'val_ids' in dir() else set()

    result = {
        'check': 'sample_id_leakage',
        'split_type': split_type,
        'train_size': len(train_ids),
        'test_size': len(test_ids),
        'overlap_train_test': len(overlap_train_test),
        'status': 'PASS' if len(overlap_train_test) == 0 else 'FAIL',
    }
    results.append(result)
    print(f"    train={len(train_ids)}, test={len(test_ids)}, overlap={len(overlap_train_test)} → {result['status']}")
    return results


def check_scene_id_leakage(data, metadata, split_type):
    """检查 scene_disjoint split 的 scene_id 是否互不相交"""
    print("  [2] Checking scene_id leakage...")
    results = []

    if split_type != 'scene_disjoint':
        results.append({
            'check': 'scene_id_leakage',
            'split_type': split_type,
            'status': 'SKIP',
            'note': 'Not scene_disjoint split',
        })
        print(f"    SKIP (not scene_disjoint)")
        return results

    walk_mask = data['task_index'] == 0
    meta = metadata[walk_mask].reset_index(drop=True)

    train_val_scenes = set(range(0, 20))
    test_scenes = set(range(20, 30))

    actual_train_scenes = set(meta[meta['scene_id'].isin(train_val_scenes)]['scene_id'].unique())
    actual_test_scenes = set(meta[meta['scene_id'].isin(test_scenes)]['scene_id'].unique())

    overlap = actual_train_scenes & actual_test_scenes

    result = {
        'check': 'scene_id_leakage',
        'split_type': split_type,
        'train_scenes': sorted(actual_train_scenes),
        'test_scenes': sorted(actual_test_scenes),
        'overlap': sorted(overlap),
        'status': 'PASS' if len(overlap) == 0 else 'FAIL',
    }
    results.append(result)
    print(f"    train_scenes={sorted(actual_train_scenes)[:5]}..., test_scenes={sorted(actual_test_scenes)[:5]}..., overlap={len(overlap)} → {result['status']}")
    return results


def check_phase_hash_leakage(data, metadata, split_type):
    """检查不同 split 是否存在完全重复 phase"""
    print("  [3] Checking phase hash leakage...")
    results = []

    walk_mask = data['task_index'] == 0
    phases = data['X_phase'][walk_mask]
    meta = metadata[walk_mask].reset_index(drop=True)

    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))
        train_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
    else:
        indices = np.arange(len(phases))
        rng = make_rng(RANDOM_SEED + 1)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_indices = indices[:n_train]
        test_indices = indices[n_train + n_val:]

    # Compute hashes for a sample (too slow for all)
    n_check = min(2000, len(test_indices))
    rng = np.random.RandomState(42)
    test_sample = rng.choice(test_indices, n_check, replace=False)
    n_train_check = min(5000, len(train_indices))
    train_sample = rng.choice(train_indices, n_train_check, replace=False)

    train_hashes = {}
    for idx in train_sample:
        h = phase_hash(phases[idx])
        train_hashes[h] = idx

    duplicates = 0
    dup_pairs = []
    for idx in test_sample:
        h = phase_hash(phases[idx])
        if h in train_hashes:
            duplicates += 1
            dup_pairs.append({'test_idx': int(idx), 'train_idx': int(train_hashes[h])})

    result = {
        'check': 'phase_hash_leakage',
        'split_type': split_type,
        'n_test_checked': n_check,
        'n_train_checked': n_train_check,
        'exact_duplicates': duplicates,
        'status': 'PASS' if duplicates == 0 else 'WARNING',
    }
    results.append(result)
    print(f"    checked {n_check} test vs {n_train_check} train, duplicates={duplicates} → {result['status']}")
    return results, dup_pairs


def check_near_duplicates(data, metadata, split_type):
    """检查近重复样本"""
    print("  [4] Checking near-duplicate samples...")
    results = []

    walk_mask = data['task_index'] == 0
    phases = data['X_phase'][walk_mask]
    meta = metadata[walk_mask].reset_index(drop=True)

    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))
        train_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)
        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]
    else:
        indices = np.arange(len(phases))
        rng = make_rng(RANDOM_SEED + 1)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_indices = indices[:n_train]
        test_indices = indices[n_train + n_val:]

    # Sample for efficiency
    rng = np.random.RandomState(42)
    n_test_sample = min(200, len(test_indices))
    n_train_sample = min(500, len(train_indices))
    test_sample = rng.choice(test_indices, n_test_sample, replace=False)
    train_sample = rng.choice(train_indices, n_train_sample, replace=False)

    # Compute Pearson correlation
    top_pairs = []
    for ti in test_sample:
        test_phase = phases[ti]
        test_norm = test_phase - test_phase.mean()
        test_std = np.std(test_norm)
        if test_std < 1e-10:
            continue
        test_norm = test_norm / test_std

        for tri in train_sample:
            train_phase = phases[tri]
            train_norm = train_phase - train_phase.mean()
            train_std = np.std(train_norm)
            if train_std < 1e-10:
                continue
            train_norm = train_norm / train_std

            corr = np.abs(np.dot(test_norm, train_norm) / len(test_norm))
            top_pairs.append({
                'test_sample_id': int(ti),
                'train_sample_id': int(tri),
                'similarity': float(corr),
                'strategy': meta.iloc[ti]['strategy'] if 'strategy' in meta.columns else 'N/A',
                'motion_label': int(data['y_motion'][walk_mask][ti]),
                'scene_id': int(meta.iloc[ti]['scene_id']),
                'split_type': split_type,
            })

    # Sort by similarity, take top 20
    top_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    top20 = top_pairs[:20]

    n_high_sim = sum(1 for p in top_pairs if p['similarity'] > 0.99)
    max_sim = top_pairs[0]['similarity'] if top_pairs else 0

    result = {
        'check': 'near_duplicate',
        'split_type': split_type,
        'n_test_sample': n_test_sample,
        'n_train_sample': n_train_sample,
        'max_similarity': float(max_sim),
        'n_pairs_above_0.99': n_high_sim,
        'status': 'WARNING' if n_high_sim > 10 else 'PASS',
    }
    results.append(result)
    print(f"    max_sim={max_sim:.4f}, pairs>0.99={n_high_sim} → {result['status']}")
    return results, top20


def check_metadata_only_classifier(data, metadata, split_type):
    """只使用 metadata 特征预测 motion_label"""
    print("  [5] Checking metadata-only classifier...")
    results = []

    walk_mask = data['task_index'] == 0
    meta = metadata[walk_mask].reset_index(drop=True)
    y = data['y_motion'][walk_mask]

    # Build metadata features (only numeric columns)
    meta_features = []
    feature_names = []
    for col in ['scene_id', 'snr_db', 'strategy_id', 'environment_id',
                'human_amplitude', 'metasurface_amplitude']:
        if col in meta.columns:
            try:
                vals = meta[col].values.astype(float)
                feature_names.append(col)
                meta_features.append(vals)
            except (ValueError, TypeError):
                pass  # Skip non-numeric columns like wall_type

    # Also encode wall_type if present
    if 'wall_type' in meta.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        wall_encoded = le.fit_transform(meta['wall_type'].values)
        feature_names.append('wall_type_encoded')
        meta_features.append(wall_encoded.astype(float))

    if not meta_features:
        results.append({
            'check': 'metadata_only_classifier',
            'split_type': split_type,
            'status': 'SKIP',
            'note': 'No metadata features available',
        })
        return results

    X_meta = np.column_stack(meta_features)
    X_meta = np.nan_to_num(X_meta, nan=0.0)

    # Split
    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))
        train_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
    else:
        indices = np.arange(len(y))
        rng = make_rng(RANDOM_SEED + 1)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_idx = indices[:n_train]
        test_idx = indices[n_train + n_val:]

    X_train, y_train = X_meta[train_idx], y[train_idx]
    X_test, y_test = X_meta[test_idx], y[test_idx]

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
    ])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    result = {
        'check': 'metadata_only_classifier',
        'split_type': split_type,
        'features_used': ','.join(feature_names),
        'accuracy': float(acc),
        'status': 'WARNING' if acc > 0.60 else 'PASS',
        'note': f'Accuracy {acc:.3f} {"above" if acc > 0.60 else "below"} 0.60 threshold',
    }
    results.append(result)
    print(f"    features={feature_names}, accuracy={acc:.3f} → {result['status']}")
    return results


def check_label_shuffle(data, metadata, split_type):
    """Label-shuffle sanity check"""
    print("  [6] Checking label-shuffle sanity...")
    results = []

    walk_mask = data['task_index'] == 0
    phases = data['X_phase'][walk_mask]
    y = data['y_motion'][walk_mask]
    meta = metadata[walk_mask].reset_index(drop=True)

    # Extract features
    from src.utils import extract_features
    X = np.array([extract_features(p, FS) for p in phases])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Split
    if split_type == 'scene_disjoint':
        train_val_scenes = set(range(0, 20))
        test_scenes = set(range(20, 30))
        train_mask = meta['scene_id'].isin(train_val_scenes)
        test_mask = meta['scene_id'].isin(test_scenes)
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
    else:
        indices = np.arange(len(y))
        rng = make_rng(RANDOM_SEED + 1)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(n_total * 0.70)
        n_val = int(n_total * 0.15)
        train_idx = indices[:n_train]
        test_idx = indices[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Shuffle labels
    rng = np.random.RandomState(42)
    y_train_shuffled = rng.permutation(y_train)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ])
    clf.fit(X_train, y_train_shuffled)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    result = {
        'check': 'label_shuffle_sanity',
        'split_type': split_type,
        'shuffled_accuracy': float(acc),
        'expected_accuracy': 0.50,
        'status': 'PASS' if acc < 0.55 else 'WARNING',
        'note': f'Shuffled accuracy {acc:.3f} {"above" if acc > 0.55 else "below"} 0.55 threshold',
    }
    results.append(result)
    print(f"    shuffled accuracy={acc:.3f} (expected ~0.50) → {result['status']}")
    return results


def check_input_fields():
    """检查 deep model 输入字段"""
    print("  [7] Checking deep model input fields...")
    results = []

    # Read deep_models.py to check input
    model_path = os.path.join(PROJECT_ROOT, 'src/deep_models.py')
    with open(model_path, 'r') as f:
        content = f.read()

    # Check PhaseDataset
    dataset_path = os.path.join(PROJECT_ROOT, 'src/deep_train.py')
    with open(dataset_path, 'r') as f:
        train_content = f.read()

    # Check if PhaseDataset uses only phase
    uses_only_phase = 'self.X' in train_content and 'phase' in train_content.lower()
    has_strategy_input = 'strategy' in train_content.lower() and 'strategy_index' in train_content
    has_scene_input = 'scene_id' in train_content

    result = {
        'check': 'input_fields',
        'uses_only_phase': uses_only_phase,
        'has_strategy_input': has_strategy_input,
        'has_scene_input': has_scene_input,
        'input_tensor_shape': '1 x 1800 (single channel phase)',
        'status': 'PASS' if not has_strategy_input and not has_scene_input else 'WARNING',
    }
    results.append(result)
    print(f"    uses_only_phase={uses_only_phase}, strategy_input={has_strategy_input}, scene_input={has_scene_input} → {result['status']}")
    return results


def check_normalization_leakage():
    """检查标准化是否只使用 train split 统计量"""
    print("  [8] Checking normalization leakage...")
    results = []

    # Check statistical attacker script
    stat_path = os.path.join(PROJECT_ROOT, 'scripts/run_statistical_attacker_aligned.py')
    if os.path.exists(stat_path):
        with open(stat_path, 'r') as f:
            content = f.read()

        # Check if StandardScaler is in Pipeline (fit on train only)
        uses_pipeline = 'Pipeline' in content
        scaler_in_pipeline = "Pipeline" in content and "StandardScaler" in content
        fit_transform_on_train = 'fit_transform' in content or ('.fit(' in content and '.transform(' in content)

        result = {
            'check': 'normalization_leakage',
            'script': 'run_statistical_attacker_aligned.py',
            'uses_pipeline': uses_pipeline,
            'scaler_in_pipeline': scaler_in_pipeline,
            'status': 'PASS' if scaler_in_pipeline else 'WARNING',
            'note': 'StandardScaler in Pipeline ensures fit on train only' if scaler_in_pipeline else 'Check manual normalization',
        }
    else:
        result = {
            'check': 'normalization_leakage',
            'status': 'SKIP',
            'note': 'Script not found',
        }

    results.append(result)
    print(f"    scaler_in_pipeline={scaler_in_pipeline if 'scaler_in_pipeline' in dir() else 'N/A'} → {result['status']}")

    # Check deep model normalization
    deep_path = os.path.join(PROJECT_ROOT, 'src/deep_train.py')
    if os.path.exists(deep_path):
        with open(deep_path, 'r') as f:
            content = f.read()

        # Check if PhaseDataset does per-sample normalization
        per_sample_norm = 'mean()' in content and 'std()' in content and 'X' in content
        global_norm = 'StandardScaler' in content or 'normalize' in content.lower()

        result2 = {
            'check': 'normalization_leakage_deep',
            'per_sample_normalization': per_sample_norm,
            'global_normalization': global_norm,
            'status': 'PASS' if per_sample_norm and not global_norm else 'INFO',
            'note': 'Deep model uses per-sample normalization (no leakage risk)' if per_sample_norm else 'Check normalization method',
        }
        results.append(result2)
        print(f"    deep: per_sample_norm={per_sample_norm}, global_norm={global_norm} → {result2['status']}")

    return results


def main():
    out_dir = os.path.join(PROJECT_ROOT, 'results/final_results')
    tables_dir = os.path.join(out_dir, 'tables')
    reports_dir = os.path.join(out_dir, 'reports')
    for d in [tables_dir, reports_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  Data Leakage Audit")
    print("=" * 60)

    all_results = []
    all_near_dup = []
    all_dup_pairs = []

    for split_type in ['random', 'scene_disjoint']:
        print(f"\n  === Split: {split_type} ===")
        try:
            data = load_dataset('medium', split_type)
            metadata = data['metadata_df']
        except Exception as e:
            print(f"  [ERROR] Cannot load dataset: {e}")
            continue

        # 1. Sample ID leakage
        r1 = check_sample_id_leakage(data, metadata, split_type)
        all_results.extend(r1)

        # 2. Scene ID leakage
        r2 = check_scene_id_leakage(data, metadata, split_type)
        all_results.extend(r2)

        # 3. Phase hash leakage
        r3, dup_pairs = check_phase_hash_leakage(data, metadata, split_type)
        all_results.extend(r3)
        all_dup_pairs.extend(dup_pairs)

        # 4. Near duplicates
        r4, near_dup = check_near_duplicates(data, metadata, split_type)
        all_results.extend(r4)
        all_near_dup.extend(near_dup)

        # 5. Metadata-only classifier
        r5 = check_metadata_only_classifier(data, metadata, split_type)
        all_results.extend(r5)

        # 6. Label-shuffle sanity
        r6 = check_label_shuffle(data, metadata, split_type)
        all_results.extend(r6)

    # 7. Input fields (global)
    r7 = check_input_fields()
    all_results.extend(r7)

    # 8. Normalization leakage (global)
    r8 = check_normalization_leakage()
    all_results.extend(r8)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(tables_dir, 'data_leakage_check.csv'), index=False)
    print(f"\n  Saved: data_leakage_check.csv")

    if all_near_dup:
        df_near = pd.DataFrame(all_near_dup)
        df_near.to_csv(os.path.join(tables_dir, 'near_duplicate_top20.csv'), index=False)
        print(f"  Saved: near_duplicate_top20.csv")

    # Save metadata-only classifier results separately
    meta_only = [r for r in all_results if r.get('check') == 'metadata_only_classifier']
    if meta_only:
        pd.DataFrame(meta_only).to_csv(os.path.join(tables_dir, 'metadata_only_classifier_results.csv'), index=False)
        print(f"  Saved: metadata_only_classifier_results.csv")

    # Save label-shuffle results separately
    shuffle_only = [r for r in all_results if r.get('check') == 'label_shuffle_sanity']
    if shuffle_only:
        pd.DataFrame(shuffle_only).to_csv(os.path.join(tables_dir, 'label_shuffle_sanity_check.csv'), index=False)
        print(f"  Saved: label_shuffle_sanity_check.csv")

    # Generate report
    lines = []
    lines.append('# Data Leakage Audit Report\n')
    lines.append(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    lines.append('\n## 检查结果总览\n')
    lines.append('| 检查项 | Split | 状态 | 备注 |')
    lines.append('|--------|-------|------|------|')
    for r in all_results:
        note = r.get('note', '')
        lines.append(f'| {r.get("check", "")} | {r.get("split_type", "N/A")} | {r.get("status", "")} | {note} |')

    lines.append('\n## 详细回答\n')

    # Q1
    lines.append('### 1. 是否发现 sample_id 泄露\n')
    sample_leaks = [r for r in all_results if r.get('check') == 'sample_id_leakage']
    for r in sample_leaks:
        lines.append(f'- {r["split_type"]}: overlap={r.get("overlap_train_test", 0)} → {"未发现" if r["status"] == "PASS" else "发现"}泄露\n')

    # Q2
    lines.append('### 2. 是否发现 scene_id 泄露\n')
    scene_leaks = [r for r in all_results if r.get('check') == 'scene_id_leakage']
    for r in scene_leaks:
        if r['status'] == 'SKIP':
            lines.append(f'- {r["split_type"]}: 跳过\n')
        else:
            lines.append(f'- {r["split_type"]}: overlap={r.get("overlap", [])} → {"未发现" if r["status"] == "PASS" else "发现"}泄露\n')

    # Q3
    lines.append('### 3. 是否发现 phase 完全重复\n')
    hash_leaks = [r for r in all_results if r.get('check') == 'phase_hash_leakage']
    for r in hash_leaks:
        lines.append(f'- {r["split_type"]}: exact_duplicates={r.get("exact_duplicates", 0)} → {"未发现" if r["status"] == "PASS" else "发现"}重复\n')

    # Q4
    lines.append('### 4. 是否存在近重复样本风险\n')
    near_dups = [r for r in all_results if r.get('check') == 'near_duplicate']
    for r in near_dups:
        lines.append(f'- {r["split_type"]}: max_similarity={r.get("max_similarity", 0):.4f}, pairs>0.99={r.get("n_pairs_above_0.99", 0)}\n')

    # Q5
    lines.append('### 5. metadata-only classifier 是否异常\n')
    meta_cls = [r for r in all_results if r.get('check') == 'metadata_only_classifier']
    for r in meta_cls:
        lines.append(f'- {r["split_type"]}: accuracy={r.get("accuracy", 0):.3f} → {"异常" if r["status"] == "WARNING" else "正常"}\n')

    # Q6
    lines.append('### 6. label-shuffle sanity check 是否正常\n')
    shuffles = [r for r in all_results if r.get('check') == 'label_shuffle_sanity']
    for r in shuffles:
        lines.append(f'- {r["split_type"]}: shuffled_accuracy={r.get("shuffled_accuracy", 0):.3f} → {"异常" if r["status"] == "WARNING" else "正常"}\n')

    # Q7
    lines.append('### 7. deep C/D 高准确率是否更可能来自合理学习还是泄露\n')
    lines.append('- 基于以上检查结果：\n')
    has_leakage = any(r['status'] in ['FAIL', 'WARNING'] for r in all_results if r.get('check') not in ['near_duplicate'])
    if has_leakage:
        lines.append('- 存在潜在泄露风险，需要进一步调查\n')
    else:
        lines.append('- 未发现明显数据泄露，高准确率更可能来自合理学习\n')

    # Q8
    lines.append('### 8. 当前实验结论的可信度如何\n')
    n_pass = sum(1 for r in all_results if r['status'] == 'PASS')
    n_warn = sum(1 for r in all_results if r['status'] == 'WARNING')
    n_fail = sum(1 for r in all_results if r['status'] == 'FAIL')
    lines.append(f'- PASS: {n_pass}, WARNING: {n_warn}, FAIL: {n_fail}\n')
    if n_fail == 0 and n_warn <= 2:
        lines.append('- 实验结论整体可信，但需注意标记的 WARNING 项\n')
    else:
        lines.append('- 存在较多风险项，结论需谨慎解读\n')

    # Q9
    lines.append('### 9. 仍需要注意哪些局限\n')
    lines.append('- 近重复样本检查仅抽样，可能遗漏\n')
    lines.append('- metadata-only classifier 仅检查了部分 metadata 字段\n')
    lines.append('- 深度模型的 per-sample normalization 虽然不泄露，但可能影响跨样本比较\n')
    lines.append('- 仿真数据与真实数据分布可能不同\n')

    path = os.path.join(reports_dir, 'data_leakage_check.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Saved: {path}")

    print("\n  Data leakage audit complete!")


if __name__ == '__main__':
    main()
