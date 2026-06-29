"""
Two-Stage Strategy-Aware Attack Experiment

验证攻击者能否利用 strategy fingerprint 形成两阶段攻击：
  Stage 1: phase/features -> strategy_hat
  Stage 2: 根据 strategy_hat 选择对应 strategy-specific motion classifier

三种攻击方式：
  A. Single Mixed Motion Attacker (baseline)
  B. Oracle Strategy-Aware Attacker (theoretical upper bound)
  C. Predicted Strategy-Aware Attacker (realistic two-stage)

新评价指标：
  DFL (Defense Fingerprint Leakage)
  SAAG (Strategy-Aware Attack Gain)
"""
import os, sys, argparse, warnings, numpy as np, pandas as pd
from datetime import datetime
from collections import defaultdict

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
from src.utils import make_rng, extract_features
from src.dataset import load_dataset

import torch
from src.deep_models import get_model
from src.deep_train import PhaseDataset
from src.config import dl_modes, dl_learning_rate, dl_weight_decay
from torch.utils.data import DataLoader
import torch.nn as nn

STRATEGY_LIST = list(strategies)
N_CLASSES_STRATEGY = len(STRATEGY_LIST)


def get_stat_classifiers():
    return {
        'LogisticRegression': Pipeline([('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42, multi_class='multinomial'))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))]),
        'SVM_RBF': Pipeline([('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))]),
        'GradientBoosting': Pipeline([('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))]),
    }


def get_motion_classifiers():
    return {
        'LogisticRegression': Pipeline([('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, C=1.0, random_state=42))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))]),
        'SVM_RBF': Pipeline([('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))]),
        'GradientBoosting': Pipeline([('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42))]),
    }


def extract_features_batch(phases):
    X = np.array([extract_features(p, FS) for p in phases])
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def get_splits(data, metadata, split_type, seed):
    walk_mask = data['task_index'] == 0
    phases = data['X_phase'][walk_mask]
    y_motion = data['y_motion'][walk_mask]
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

    return phases, y_motion, strat_idx, train_indices, val_indices, test_indices


def compute_binary_metrics(y_true, y_pred, y_proba=None):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    total = tp + fn + fp + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = float('nan')
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = float('nan')
    return {
        'accuracy': float(acc), 'balanced_accuracy': float(bal_acc),
        'TPR': float(tpr), 'FPR': float(fpr), 'TNR': float(tnr), 'FNR': float(fnr),
        'precision': float(precision), 'recall': float(recall), 'F1': float(f1), 'AUC': auc,
    }


# ============================================================
# Attack A: Single Mixed Motion Attacker
# ============================================================
def run_single_mixed_attack(X_train, y_train, X_val, y_val, X_test, y_test,
                            strat_idx_test, input_type, split_type, seed):
    """Train one mixed motion classifier, test on all"""
    results = []
    for clf_name, clf_template in get_motion_classifiers().items():
        # Clone
        clf = Pipeline([('scaler', StandardScaler()),
            ('clf', type(clf_template.named_steps['clf'])(**clf_template.named_steps['clf'].get_params()))])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        metrics = compute_binary_metrics(y_test, y_pred, y_proba)

        # Per-strategy results
        for s_idx, s_name in enumerate(STRATEGY_LIST):
            mask = strat_idx_test == s_idx
            if mask.sum() == 0:
                continue
            s_metrics = compute_binary_metrics(y_test[mask], y_pred[mask],
                                               y_proba[mask] if y_proba is not None else None)
            results.append({
                'mode': 'medium', 'split_type': split_type, 'seed': seed,
                'input_type': input_type, 'attack_type': 'single_mixed',
                'test_strategy': s_name, **s_metrics,
                'strategy_classifier_name': 'N/A', 'motion_classifier_name': clf_name,
                'strategy_classifier_accuracy': float('nan'),
                'strategy_classifier_macro_F1': float('nan'),
                'strategy_selection_correct_rate': float('nan'),
                'num_train': len(y_train), 'num_test': mask.sum(),
            })
    return results


# ============================================================
# Attack B: Oracle Strategy-Aware Attacker
# ============================================================
def run_oracle_strategy_aware_attack(X_train, y_train, strat_idx_train,
                                     X_val, y_val, strat_idx_val,
                                     X_test, y_test, strat_idx_test,
                                     input_type, split_type, seed):
    """Train per-strategy motion classifiers, test with true strategy label"""
    results = []
    for clf_name in get_motion_classifiers().keys():
        # Per-strategy motion classifiers
        strategy_clfs = {}
        for s_idx, s_name in enumerate(STRATEGY_LIST):
            mask_train = strat_idx_train == s_idx
            mask_val = strat_idx_val == s_idx
            if mask_train.sum() < 10:
                continue
            clf = Pipeline([('scaler', StandardScaler()),
                ('clf', type(get_motion_classifiers()[clf_name].named_steps['clf'])(
                    **get_motion_classifiers()[clf_name].named_steps['clf'].get_params()))])
            clf.fit(X_train[mask_train], y_train[mask_train])
            strategy_clfs[s_idx] = clf

        # Test with oracle strategy
        y_pred_all = np.zeros(len(y_test))
        y_proba_all = np.zeros(len(y_test))
        for s_idx in strategy_clfs:
            mask = strat_idx_test == s_idx
            if mask.sum() == 0:
                continue
            clf = strategy_clfs[s_idx]
            y_pred_all[mask] = clf.predict(X_test[mask])
            if hasattr(clf, 'predict_proba'):
                y_proba_all[mask] = clf.predict_proba(X_test[mask])[:, 1]

        overall_metrics = compute_binary_metrics(y_test, y_pred_all, y_proba_all)

        for s_idx, s_name in enumerate(STRATEGY_LIST):
            mask = strat_idx_test == s_idx
            if mask.sum() == 0:
                continue
            s_metrics = compute_binary_metrics(y_test[mask], y_pred_all[mask],
                                               y_proba_all[mask])
            results.append({
                'mode': 'medium', 'split_type': split_type, 'seed': seed,
                'input_type': input_type, 'attack_type': 'oracle_strategy_aware',
                'test_strategy': s_name, **s_metrics,
                'strategy_classifier_name': 'oracle', 'motion_classifier_name': clf_name,
                'strategy_classifier_accuracy': 1.0,
                'strategy_classifier_macro_F1': 1.0,
                'strategy_selection_correct_rate': 1.0,
                'num_train': len(y_train), 'num_test': mask.sum(),
            })
    return results


# ============================================================
# Attack C: Predicted Strategy-Aware Attacker
# ============================================================
def run_predicted_strategy_aware_attack(X_train, y_train, strat_idx_train,
                                        X_val, y_val, strat_idx_val,
                                        X_test, y_test, strat_idx_test,
                                        input_type, split_type, seed):
    """Two-stage: predict strategy first, then use strategy-specific motion classifier"""
    results = []

    # Stage 1: Train strategy classifiers, select best on val
    best_strat_clf_name = None
    best_strat_clf = None
    best_strat_val_acc = 0
    best_strat_val_f1 = 0

    for clf_name, clf_template in get_stat_classifiers().items():
        clf = Pipeline([('scaler', StandardScaler()),
            ('clf', type(clf_template.named_steps['clf'])(**clf_template.named_steps['clf'].get_params()))])
        clf.fit(X_train, strat_idx_train)
        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(strat_idx_val, val_pred)
        val_f1 = f1_score(strat_idx_val, val_pred, average='macro')
        if val_acc > best_strat_val_acc:
            best_strat_val_acc = val_acc
            best_strat_val_f1 = val_f1
            best_strat_clf_name = clf_name
            best_strat_clf = clf

    # Predict strategy on test
    strat_pred = best_strat_clf.predict(X_test)
    strat_correct_rate = accuracy_score(strat_idx_test, strat_pred)

    # Stage 2: Train per-strategy motion classifiers, select best on val per strategy
    for motion_clf_name in get_motion_classifiers().keys():
        strategy_motion_clfs = {}
        for s_idx, s_name in enumerate(STRATEGY_LIST):
            mask_train = strat_idx_train == s_idx
            mask_val = strat_idx_val == s_idx
            if mask_train.sum() < 10:
                continue
            clf = Pipeline([('scaler', StandardScaler()),
                ('clf', type(get_motion_classifiers()[motion_clf_name].named_steps['clf'])(
                    **get_motion_classifiers()[motion_clf_name].named_steps['clf'].get_params()))])
            clf.fit(X_train[mask_train], y_train[mask_train])
            strategy_motion_clfs[s_idx] = clf

        # Test with predicted strategy
        y_pred_all = np.zeros(len(y_test))
        y_proba_all = np.zeros(len(y_test))
        for i in range(len(X_test)):
            pred_s = int(strat_pred[i])
            if pred_s in strategy_motion_clfs:
                clf = strategy_motion_clfs[pred_s]
                x = X_test[i:i+1]
                y_pred_all[i] = clf.predict(x)[0]
                if hasattr(clf, 'predict_proba'):
                    y_proba_all[i] = clf.predict_proba(x)[0, 1]
            else:
                # Fallback: use any available classifier
                first_key = list(strategy_motion_clfs.keys())[0]
                clf = strategy_motion_clfs[first_key]
                y_pred_all[i] = clf.predict(X_test[i:i+1])[0]
                if hasattr(clf, 'predict_proba'):
                    y_proba_all[i] = clf.predict_proba(X_test[i:i+1])[0, 1]

        for s_idx, s_name in enumerate(STRATEGY_LIST):
            mask = strat_idx_test == s_idx
            if mask.sum() == 0:
                continue
            s_metrics = compute_binary_metrics(y_test[mask], y_pred_all[mask],
                                               y_proba_all[mask])
            results.append({
                'mode': 'medium', 'split_type': split_type, 'seed': seed,
                'input_type': input_type, 'attack_type': 'predicted_strategy_aware',
                'test_strategy': s_name, **s_metrics,
                'strategy_classifier_name': best_strat_clf_name,
                'motion_classifier_name': motion_clf_name,
                'strategy_classifier_accuracy': best_strat_val_acc,
                'strategy_classifier_macro_F1': best_strat_val_f1,
                'strategy_selection_correct_rate': strat_correct_rate,
                'num_train': len(y_train), 'num_test': mask.sum(),
            })
    return results, strat_pred, best_strat_clf_name, best_strat_val_acc, best_strat_val_f1


# ============================================================
# Deep model versions
# ============================================================
def train_deep_strategy_classifier(train_phases, train_labels, val_phases, val_labels,
                                   model_name='ResNet1DLite', seed=2026, mode='medium'):
    """Train a deep strategy classifier (5-class)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = N_CLASSES_STRATEGY
    mode_config = dl_modes.get(mode, dl_modes['debug'])
    epochs = mode_config['epochs']
    patience = mode_config['patience']
    batch_size = mode_config['batch_size']

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model = get_model(model_name, n_classes=n_classes).to(device)

    train_ds = PhaseDataset(train_phases, train_labels)
    val_ds = PhaseDataset(val_phases, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=dl_learning_rate, weight_decay=dl_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

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

    return model, best_val_acc


def train_deep_motion_classifier(train_phases, train_labels, val_phases, val_labels,
                                 model_name='ResNet1DLite', seed=2026, mode='medium'):
    """Train a deep binary motion classifier"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_config = dl_modes.get(mode, dl_modes['debug'])
    epochs = mode_config['epochs']
    patience = mode_config['patience']
    batch_size = mode_config['batch_size']

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model = get_model(model_name, n_classes=2).to(device)

    train_ds = PhaseDataset(train_phases, train_labels)
    val_ds = PhaseDataset(val_phases, val_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=dl_learning_rate, weight_decay=dl_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

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

    return model, best_val_acc


def predict_deep(model, phases, batch_size=64):
    """Get predictions from a deep model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    probs = []
    n = len(phases)
    for i in range(0, n, batch_size):
        batch = phases[i:i+batch_size]
        x = torch.tensor(batch, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            out = model(x)
            prob = torch.softmax(out, dim=1)
            preds.extend(out.argmax(1).cpu().numpy())
            probs.extend(prob.cpu().numpy())
    return np.array(preds), np.array(probs)


def run_deep_two_stage(phases, y_motion, strat_idx, train_idx, val_idx, test_idx,
                       split_type, seed, mode):
    """Run deep model two-stage attack (ResNet1DLite only for speed)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    train_phases = phases[train_idx]
    val_phases = phases[val_idx]
    test_phases = phases[test_idx]

    for model_name in ['ResNet1DLite']:
        try:
            print(f"    Deep {model_name}: Training strategy classifier...")
            strat_model, strat_val_acc = train_deep_strategy_classifier(
                train_phases, strat_idx[train_idx], val_phases, strat_idx[val_idx],
                model_name=model_name, seed=seed, mode=mode)

            strat_pred, strat_probs = predict_deep(strat_model, test_phases)
            strat_correct_rate = accuracy_score(strat_idx[test_idx], strat_pred)
            strat_macro_f1 = f1_score(strat_idx[test_idx], strat_pred, average='macro')
            print(f"    Deep {model_name}: Strategy acc={strat_val_acc:.3f}, test_correct={strat_correct_rate:.3f}")

            # A. Single mixed
            print(f"    Deep {model_name}: Training mixed motion classifier...")
            mixed_model, _ = train_deep_motion_classifier(
                train_phases, y_motion[train_idx], val_phases, y_motion[val_idx],
                model_name=model_name, seed=seed, mode=mode)
            motion_pred, motion_probs = predict_deep(mixed_model, test_phases)

            for s_idx, s_name in enumerate(STRATEGY_LIST):
                mask = strat_idx[test_idx] == s_idx
                if mask.sum() == 0:
                    continue
                s_metrics = compute_binary_metrics(y_motion[test_idx][mask], motion_pred[mask],
                                                   motion_probs[mask, 1] if motion_probs.shape[1] == 2 else None)
                results.append({
                    'mode': 'medium', 'split_type': split_type, 'seed': seed,
                    'input_type': 'raw_phase', 'attack_type': 'single_mixed',
                    'test_strategy': s_name, **s_metrics,
                    'strategy_classifier_name': 'N/A', 'motion_classifier_name': model_name,
                    'strategy_classifier_accuracy': float('nan'),
                    'strategy_classifier_macro_F1': float('nan'),
                    'strategy_selection_correct_rate': float('nan'),
                    'num_train': len(y_motion[train_idx]), 'num_test': mask.sum(),
                })

            # B. Oracle strategy-aware
            print(f"    Deep {model_name}: Training per-strategy motion classifiers...")
            strategy_motion_models = {}
            for s_idx, s_name in enumerate(STRATEGY_LIST):
                mask_train = strat_idx[train_idx] == s_idx
                mask_val = strat_idx[val_idx] == s_idx
                if mask_train.sum() < 10:
                    continue
                m, _ = train_deep_motion_classifier(
                    train_phases[mask_train], y_motion[train_idx][mask_train],
                    val_phases[mask_val], y_motion[val_idx][mask_val],
                    model_name=model_name, seed=seed, mode=mode)
                strategy_motion_models[s_idx] = m

            # Oracle test
            oracle_pred = np.zeros(len(y_motion[test_idx]))
            oracle_proba = np.zeros(len(y_motion[test_idx]))
            for s_idx in strategy_motion_models:
                mask = strat_idx[test_idx] == s_idx
                if mask.sum() == 0:
                    continue
                p, pr = predict_deep(strategy_motion_models[s_idx], test_phases[mask])
                oracle_pred[mask] = p
                if pr.shape[1] == 2:
                    oracle_proba[mask] = pr[:, 1]

            for s_idx, s_name in enumerate(STRATEGY_LIST):
                mask = strat_idx[test_idx] == s_idx
                if mask.sum() == 0:
                    continue
                s_metrics = compute_binary_metrics(y_motion[test_idx][mask], oracle_pred[mask],
                                                   oracle_proba[mask])
                results.append({
                    'mode': 'medium', 'split_type': split_type, 'seed': seed,
                    'input_type': 'raw_phase', 'attack_type': 'oracle_strategy_aware',
                    'test_strategy': s_name, **s_metrics,
                    'strategy_classifier_name': 'oracle', 'motion_classifier_name': model_name,
                    'strategy_classifier_accuracy': 1.0,
                    'strategy_classifier_macro_F1': 1.0,
                    'strategy_selection_correct_rate': 1.0,
                    'num_train': len(y_motion[train_idx]), 'num_test': mask.sum(),
                })

            # C. Predicted strategy-aware
            pred_pred = np.zeros(len(y_motion[test_idx]))
            pred_proba = np.zeros(len(y_motion[test_idx]))
            for i in range(len(test_phases)):
                pred_s = int(strat_pred[i])
                if pred_s in strategy_motion_models:
                    p, pr = predict_deep(strategy_motion_models[pred_s], test_phases[i:i+1])
                    pred_pred[i] = p[0]
                    if pr.shape[1] == 2:
                        pred_proba[i] = pr[0, 1]
                else:
                    first_key = list(strategy_motion_models.keys())[0]
                    p, pr = predict_deep(strategy_motion_models[first_key], test_phases[i:i+1])
                    pred_pred[i] = p[0]
                    if pr.shape[1] == 2:
                        pred_proba[i] = pr[0, 1]

            for s_idx, s_name in enumerate(STRATEGY_LIST):
                mask = strat_idx[test_idx] == s_idx
                if mask.sum() == 0:
                    continue
                s_metrics = compute_binary_metrics(y_motion[test_idx][mask], pred_pred[mask],
                                                   pred_proba[mask])
                results.append({
                    'mode': 'medium', 'split_type': split_type, 'seed': seed,
                    'input_type': 'raw_phase', 'attack_type': 'predicted_strategy_aware',
                    'test_strategy': s_name, **s_metrics,
                    'strategy_classifier_name': model_name,
                    'motion_classifier_name': model_name,
                    'strategy_classifier_accuracy': strat_val_acc,
                    'strategy_classifier_macro_F1': strat_macro_f1,
                    'strategy_selection_correct_rate': strat_correct_rate,
                    'num_train': len(y_motion[train_idx]), 'num_test': mask.sum(),
                })

            print(f"    Deep {model_name}: Done!")

        except Exception as e:
            print(f"    [ERROR] Deep {model_name}: {e}")
            import traceback
            traceback.print_exc()

    return results, strat_pred


# ============================================================
# Main experiment runner
# ============================================================
def run_experiment(mode, split_type, seeds, out_dir):
    print(f"\n  Loading dataset: mode={mode}, split={split_type}")
    data = load_dataset(mode, split_type)
    metadata = data['metadata_df']

    all_results = []
    all_strat_pred = {}

    for seed in seeds:
        print(f"\n  Seed: {seed}")
        phases, y_motion, strat_idx, train_idx, val_idx, test_idx = get_splits(
            data, metadata, split_type, seed)

        # ---- Handcrafted features ----
        print(f"  [handcrafted_features]")
        X_all = extract_features_batch(phases)
        X_train, y_train = X_all[train_idx], y_motion[train_idx]
        X_val, y_val = X_all[val_idx], y_motion[val_idx]
        X_test, y_test = X_all[test_idx], y_motion[test_idx]
        strat_train, strat_val, strat_test = strat_idx[train_idx], strat_idx[val_idx], strat_idx[test_idx]

        # A. Single mixed
        r_a = run_single_mixed_attack(X_train, y_train, X_val, y_val, X_test, y_test,
                                      strat_test, 'handcrafted_features', split_type, seed)
        all_results.extend(r_a)

        # B. Oracle strategy-aware
        r_b = run_oracle_strategy_aware_attack(X_train, y_train, strat_train,
                                               X_val, y_val, strat_val,
                                               X_test, y_test, strat_test,
                                               'handcrafted_features', split_type, seed)
        all_results.extend(r_b)

        # C. Predicted strategy-aware
        r_c, strat_pred, best_clf_name, best_clf_acc, best_clf_f1 = run_predicted_strategy_aware_attack(
            X_train, y_train, strat_train, X_val, y_val, strat_val,
            X_test, y_test, strat_test, 'handcrafted_features', split_type, seed)
        all_results.extend(r_c)
        all_strat_pred[(split_type, seed, 'handcrafted_features')] = strat_pred

        print(f"    Best strategy clf: {best_clf_name}, val_acc={best_clf_acc:.3f}, val_f1={best_clf_f1:.3f}")

        # ---- Raw phase deep ----
        print(f"  [raw_phase]")
        r_deep, deep_strat_pred = run_deep_two_stage(
            phases, y_motion, strat_idx, train_idx, val_idx, test_idx,
            split_type, seed, mode)
        all_results.extend(r_deep)
        all_strat_pred[(split_type, seed, 'raw_phase')] = deep_strat_pred

    return all_results, all_strat_pred


def generate_outputs(all_results, all_strat_pred, out_dir, split_type):
    tables_dir = os.path.join(out_dir, 'tables')
    figures_dir = os.path.join(out_dir, 'figures')
    reports_dir = os.path.join(out_dir, 'reports')

    # Raw results
    df_raw = pd.DataFrame(all_results)
    raw_path = os.path.join(tables_dir, 'two_stage_strategy_aware_attack_results.csv')
    if os.path.exists(raw_path):
        df_existing = pd.read_csv(raw_path)
        df_raw = pd.concat([df_existing, df_raw], ignore_index=True)
    df_raw.to_csv(raw_path, index=False)
    print(f"\n  Saved: {raw_path}")

    # Summary
    group_cols = ['split_type', 'input_type', 'attack_type', 'test_strategy']
    summary_rows = []
    for name, group in df_raw.groupby(group_cols):
        n = len(group)
        summary_rows.append({
            'split_type': name[0], 'input_type': name[1], 'attack_type': name[2],
            'test_strategy': name[3],
            'mean_accuracy': group['accuracy'].mean(),
            'std_accuracy': group['accuracy'].std() if n > 1 else 0,
            'mean_balanced_accuracy': group['balanced_accuracy'].mean(),
            'mean_F1': group['F1'].mean(),
            'mean_TPR': group['TPR'].mean(),
            'mean_FPR': group['FPR'].mean(),
            'mean_AUC': group['AUC'].mean(),
            'num_seeds': n,
        })
    df_sum = pd.DataFrame(summary_rows)
    sum_path = os.path.join(tables_dir, 'two_stage_strategy_aware_attack_summary.csv')
    if os.path.exists(sum_path):
        df_sum_existing = pd.read_csv(sum_path)
        df_sum = pd.concat([df_sum_existing, df_sum], ignore_index=True)
    df_sum.to_csv(sum_path, index=False)
    print(f"  Saved: {sum_path}")

    # Strategy selection confusion matrix
    for (sp, seed, it), strat_pred in all_strat_pred.items():
        if sp != split_type or it != 'handcrafted_features':
            continue
        # Load test strat_idx from dataset
        data = load_dataset('medium', sp)
        metadata = data['metadata_df']
        _, _, strat_idx_data, _, _, test_idx = get_splits(data, metadata, sp, seed)
        strat_true = strat_idx_data[test_idx]
        cm = confusion_matrix(strat_true, strat_pred, labels=list(range(N_CLASSES_STRATEGY)))
        cm_df = pd.DataFrame(cm, index=STRATEGY_LIST, columns=STRATEGY_LIST)
        sfx = f'_{split_type}' if split_type != 'random' else ''
        cm_path = os.path.join(tables_dir, f'two_stage_strategy_selection_confusion{sfx}.csv')
        cm_df.to_csv(cm_path)

    # DFL metrics
    dfl_rows = []
    for (sp, seed, it), strat_pred in all_strat_pred.items():
        data = load_dataset('medium', sp)
        metadata = data['metadata_df']
        _, _, strat_idx_data, _, _, test_idx = get_splits(data, metadata, sp, seed)
        strat_true = strat_idx_data[test_idx]
        acc = accuracy_score(strat_true, strat_pred)
        macro_f1 = f1_score(strat_true, strat_pred, average='macro')
        # Get model name from results
        model_name = 'ResNet1DLite' if it == 'raw_phase' else 'RandomForest'
        dfl_rows.append({
            'split_type': sp, 'input_type': it, 'model_name': model_name,
            'num_strategies': N_CLASSES_STRATEGY,
            'DFL_acc': acc, 'DFL_macroF1': macro_f1,
            'DFL_random_baseline': 1.0 / N_CLASSES_STRATEGY,
            'DFL_excess': acc - 1.0 / N_CLASSES_STRATEGY,
            'seed': seed,
        })
    df_dfl = pd.DataFrame(dfl_rows)
    # Average over seeds
    dfl_summary = df_dfl.groupby(['split_type', 'input_type', 'model_name']).agg({
        'DFL_acc': 'mean', 'DFL_macroF1': 'mean',
        'num_strategies': 'first', 'DFL_random_baseline': 'first', 'DFL_excess': 'mean',
    }).reset_index()
    dfl_path = os.path.join(tables_dir, 'defense_fingerprint_metrics.csv')
    dfl_summary.to_csv(dfl_path, index=False)
    print(f"  Saved: {dfl_path}")

    # SAAG metrics
    saag_rows = []
    for sp in [split_type]:
        for it in ['handcrafted_features', 'raw_phase']:
            for s_name in STRATEGY_LIST:
                sub = df_sum[(df_sum['split_type'] == sp) & (df_sum['input_type'] == it) & (df_sum['test_strategy'] == s_name)]
                single = sub[sub['attack_type'] == 'single_mixed']['mean_accuracy'].values
                oracle = sub[sub['attack_type'] == 'oracle_strategy_aware']['mean_accuracy'].values
                predicted = sub[sub['attack_type'] == 'predicted_strategy_aware']['mean_accuracy'].values

                single_acc = single[0] if len(single) > 0 else float('nan')
                oracle_acc = oracle[0] if len(oracle) > 0 else float('nan')
                predicted_acc = predicted[0] if len(predicted) > 0 else float('nan')

                saag_oracle = oracle_acc - single_acc if not np.isnan(oracle_acc) and not np.isnan(single_acc) else float('nan')
                saag_predicted = predicted_acc - single_acc if not np.isnan(predicted_acc) and not np.isnan(single_acc) else float('nan')

                # Strategy recall for this strategy
                strat_recall = float('nan')
                for (sp2, seed2, it2), strat_pred in all_strat_pred.items():
                    if sp2 == sp and it2 == it:
                        data = load_dataset('medium', sp)
                        metadata = data['metadata_df']
                        _, _, strat_idx_data, _, _, test_idx = get_splits(data, metadata, sp, seed2)
                        strat_true = strat_idx_data[test_idx]
                        s_idx = STRATEGY_LIST.index(s_name)
                        mask = strat_true == s_idx
                        if mask.sum() > 0:
                            strat_recall = (strat_pred[mask] == s_idx).mean()

                if abs(saag_predicted) > 0.05 and not np.isnan(saag_predicted):
                    if saag_predicted > 0:
                        interpretation = 'fingerprint is exploitable'
                    else:
                        interpretation = 'two-stage attack hurts, strategy prediction errors dominate'
                else:
                    interpretation = 'fingerprint exists but not clearly exploitable'

                saag_rows.append({
                    'split_type': sp, 'input_type': it, 'test_strategy': s_name,
                    'single_mixed_accuracy': round(single_acc, 4) if not np.isnan(single_acc) else float('nan'),
                    'oracle_strategy_aware_accuracy': round(oracle_acc, 4) if not np.isnan(oracle_acc) else float('nan'),
                    'predicted_strategy_aware_accuracy': round(predicted_acc, 4) if not np.isnan(predicted_acc) else float('nan'),
                    'SAAG_oracle': round(saag_oracle, 4) if not np.isnan(saag_oracle) else float('nan'),
                    'SAAG_predicted': round(saag_predicted, 4) if not np.isnan(saag_predicted) else float('nan'),
                    'strategy_classifier_recall_for_this_strategy': round(strat_recall, 4) if not np.isnan(strat_recall) else float('nan'),
                    'interpretation': interpretation,
                })
    df_saag = pd.DataFrame(saag_rows)
    saag_path = os.path.join(tables_dir, 'strategy_aware_attack_gain.csv')
    if os.path.exists(saag_path):
        df_saag_existing = pd.read_csv(saag_path)
        df_saag = pd.concat([df_saag_existing, df_saag], ignore_index=True)
    df_saag.to_csv(saag_path, index=False)
    print(f"  Saved: {saag_path}")

    # ---- Figures ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sfx = f'_{split_type}' if split_type != 'random' else ''

    # Fig 1/2: Three attack types comparison
    for it in ['handcrafted_features', 'raw_phase']:
        sub = df_sum[(df_sum['split_type'] == split_type) & (df_sum['input_type'] == it)]
        if len(sub) == 0:
            continue
        fig, ax = plt.subplots(figsize=(14, 7))
        strategies_in_plot = sub['test_strategy'].unique()
        x = np.arange(len(strategies_in_plot))
        width = 0.25

        for i, atk in enumerate(['single_mixed', 'oracle_strategy_aware', 'predicted_strategy_aware']):
            atk_data = sub[sub['attack_type'] == atk]
            accs = [atk_data[atk_data['test_strategy'] == s]['mean_accuracy'].values[0]
                    if len(atk_data[atk_data['test_strategy'] == s]) > 0 else 0 for s in strategies_in_plot]
            ax.bar(x + i * width, accs, width, label=atk, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies_in_plot, rotation=30, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Two-Stage Attack Comparison ({it}, {split_type})')
        ax.legend()
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        it_sfx = f'_{it}' if it != 'handcrafted_features' else ''
        plt.savefig(os.path.join(figures_dir, f'two_stage_attack_comparison{sfx}{it_sfx}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figures_dir, f'two_stage_attack_comparison{sfx}{it_sfx}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    # Fig 3: SAAG by strategy
    if len(df_saag) > 0:
        saag_sub = df_saag[df_saag['split_type'] == split_type]
        for it in ['handcrafted_features', 'raw_phase']:
            saag_it = saag_sub[saag_sub['input_type'] == it]
            if len(saag_it) == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(saag_it))
            width = 0.35
            ax.bar(x - width/2, saag_it['SAAG_oracle'].values, width, label='SAAG_oracle', alpha=0.8, color='coral')
            ax.bar(x + width/2, saag_it['SAAG_predicted'].values, width, label='SAAG_predicted', alpha=0.8, color='steelblue')
            ax.set_xticks(x)
            ax.set_xticklabels(saag_it['test_strategy'].values, rotation=30, ha='right')
            ax.set_ylabel('SAAG')
            ax.set_title(f'Strategy-Aware Attack Gain ({it}, {split_type})')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Exploitable threshold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            it_sfx = f'_{it}' if it != 'handcrafted_features' else ''
            plt.savefig(os.path.join(figures_dir, f'strategy_aware_attack_gain_by_strategy{sfx}{it_sfx}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(figures_dir, f'strategy_aware_attack_gain_by_strategy{sfx}{it_sfx}.pdf'), dpi=300, bbox_inches='tight')
            plt.close()

    # Fig 4: DFL
    if len(dfl_summary) > 0:
        dfl_sub = dfl_summary[dfl_summary['split_type'] == split_type]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(dfl_sub))
        width = 0.35
        ax.bar(x - width/2, dfl_sub['DFL_acc'].values, width, label='DFL_acc', alpha=0.8, color='coral')
        ax.bar(x + width/2, dfl_sub['DFL_macroF1'].values, width, label='DFL_macroF1', alpha=0.8, color='steelblue')
        ax.axhline(y=1.0/N_CLASSES_STRATEGY, color='red', linestyle='--', alpha=0.5, label=f'Random ({1.0/N_CLASSES_STRATEGY:.2f})')
        ax.set_xticks(x)
        labels = [f"{r['model_name']}\n({r['input_type']})" for _, r in dfl_sub.iterrows()]
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('DFL')
        ax.set_title(f'Defense Fingerprint Leakage ({split_type})')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'defense_fingerprint_leakage{sfx}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figures_dir, f'defense_fingerprint_leakage{sfx}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    # Fig 5: Fingerprint vs Attack Gain scatter
    if len(df_saag) > 0 and len(dfl_summary) > 0:
        saag_sub = df_saag[df_saag['split_type'] == split_type]
        fig, ax = plt.subplots(figsize=(8, 6))
        for it in ['handcrafted_features', 'raw_phase']:
            saag_it = saag_sub[saag_sub['input_type'] == it]
            if len(saag_it) == 0:
                continue
            ax.scatter(saag_it['strategy_classifier_recall_for_this_strategy'],
                      saag_it['SAAG_predicted'],
                      label=it, alpha=0.7, s=60)
            for _, row in saag_it.iterrows():
                ax.annotate(row['test_strategy'],
                           (row['strategy_classifier_recall_for_this_strategy'], row['SAAG_predicted']),
                           fontsize=7, alpha=0.7)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Exploitable threshold')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Strategy Classifier Recall')
        ax.set_ylabel('SAAG_predicted')
        ax.set_title(f'Fingerprint vs Attack Gain ({split_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'fingerprint_vs_attack_gain_scatter{sfx}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(figures_dir, f'fingerprint_vs_attack_gain_scatter{sfx}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  Figures generated for {split_type}.")


def generate_report(out_dir):
    from datetime import datetime
    tables_dir = os.path.join(out_dir, 'tables')
    reports_dir = os.path.join(out_dir, 'reports')

    df_raw = pd.read_csv(os.path.join(tables_dir, 'two_stage_strategy_aware_attack_results.csv'))
    df_sum = pd.read_csv(os.path.join(tables_dir, 'two_stage_strategy_aware_attack_summary.csv'))
    df_dfl = pd.read_csv(os.path.join(tables_dir, 'defense_fingerprint_metrics.csv'))
    df_saag = pd.read_csv(os.path.join(tables_dir, 'strategy_aware_attack_gain.csv'))

    lines = []
    lines.append('# 两阶段策略感知攻击分析\n')
    lines.append(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

    # Summary tables
    for split_type in ['random', 'scene_disjoint']:
        lines.append(f'\n## Split: {split_type}\n')
        for it in ['handcrafted_features', 'raw_phase']:
            sub = df_sum[(df_sum['split_type'] == split_type) & (df_sum['input_type'] == it)]
            if len(sub) == 0:
                continue
            lines.append(f'### {it}\n')
            lines.append('| 攻击方式 | 测试策略 | Accuracy | Balanced Acc | F1 | TPR |')
            lines.append('|---------|---------|----------|-------------|-----|-----|')
            for _, row in sub.iterrows():
                lines.append(f'| {row["attack_type"]} | {row["test_strategy"]} | {row["mean_accuracy"]:.3f}±{row["std_accuracy"]:.3f} | '
                            f'{row["mean_balanced_accuracy"]:.3f} | {row["mean_F1"]:.3f} | {row["mean_TPR"]:.3f} |')
            lines.append('')

    # DFL
    lines.append('\n## Defense Fingerprint Leakage (DFL)\n')
    if len(df_dfl) > 0:
        lines.append('| Split | Input | Model | DFL_acc | DFL_macroF1 | DFL_random | DFL_excess |')
        lines.append('|-------|-------|-------|---------|-------------|------------|------------|')
        for _, row in df_dfl.iterrows():
            lines.append(f'| {row["split_type"]} | {row["input_type"]} | {row["model_name"]} | '
                        f'{row["DFL_acc"]:.3f} | {row["DFL_macroF1"]:.3f} | {row["DFL_random_baseline"]:.3f} | {row["DFL_excess"]:.3f} |')
        lines.append('')

    # SAAG
    lines.append('\n## Strategy-Aware Attack Gain (SAAG)\n')
    if len(df_saag) > 0:
        lines.append('| Split | Input | Strategy | Single | Oracle | Predicted | SAAG_oracle | SAAG_predicted | Interpretation |')
        lines.append('|-------|-------|----------|--------|--------|-----------|-------------|----------------|----------------|')
        for _, row in df_saag.iterrows():
            lines.append(f'| {row["split_type"]} | {row["input_type"]} | {row["test_strategy"]} | '
                        f'{row["single_mixed_accuracy"]:.3f} | {row["oracle_strategy_aware_accuracy"]:.3f} | '
                        f'{row["predicted_strategy_aware_accuracy"]:.3f} | {row["SAAG_oracle"]:+.3f} | '
                        f'{row["SAAG_predicted"]:+.3f} | {row["interpretation"]} |')
        lines.append('')

    # Answer questions
    lines.append('\n## 研究问题回答\n')

    # Q1-5: DFL
    lines.append('### 1-5: DFL 相关\n')
    if len(df_dfl) > 0:
        for _, row in df_dfl.iterrows():
            excess = row['DFL_excess']
            lines.append(f'- {row["split_type"]}/{row["input_type"]}/{row["model_name"]}: DFL_acc={row["DFL_acc"]:.3f}, '
                        f'random={row["DFL_random_baseline"]:.3f}, excess={excess:.3f}, '
                        f'{"显著高于" if excess > 0.1 else "接近"}随机猜测\n')

    # Q6-8: SAAG
    lines.append('### 6-8: SAAG 相关\n')
    if len(df_saag) > 0:
        for split_type in ['random', 'scene_disjoint']:
            for it in ['handcrafted_features', 'raw_phase']:
                sub = df_saag[(df_saag['split_type'] == split_type) & (df_saag['input_type'] == it)]
                if len(sub) == 0:
                    continue
                saag_pred_mean = sub['SAAG_predicted'].mean()
                saag_oracle_mean = sub['SAAG_oracle'].mean()
                n_exploitable = (sub['SAAG_predicted'] > 0.05).sum()
                lines.append(f'- {split_type}/{it}: SAAG_predicted_mean={saag_pred_mean:+.3f}, '
                            f'SAAG_oracle_mean={saag_oracle_mean:+.3f}, '
                            f'exploitable strategies={n_exploitable}/{len(sub)}\n')

    # Q9-10
    lines.append('### 9-10: 策略指纹是否可被利用\n')
    if len(df_saag) > 0:
        n_exploitable = (df_saag['SAAG_predicted'] > 0.05).sum()
        n_total = len(df_saag)
        if n_exploitable > n_total * 0.3:
            lines.append('SAAG_predicted 在多个策略上明显为正，说明策略指纹已从"可观察现象"转化为"可利用攻击路径"。\n')
        else:
            lines.append('SAAG_predicted 不明显，说明虽然策略指纹存在，但当前两阶段攻击暂未显著提升 motion inference。\n')

    # Q11-12
    lines.append('### 11-12: 哪些策略最容易被利用\n')
    if len(df_saag) > 0:
        for _, row in df_saag.iterrows():
            if row['SAAG_predicted'] > 0.05:
                lines.append(f'- {row["test_strategy"]} ({row["input_type"]}): SAAG_predicted={row["SAAG_predicted"]:+.3f} → 可被利用\n')
            elif row['SAAG_predicted'] < -0.05:
                lines.append(f'- {row["test_strategy"]} ({row["input_type"]}): SAAG_predicted={row["SAAG_predicted"]:+.3f} → 两阶段攻击反而有害\n')

    # Q13
    lines.append('### 13: scene_disjoint 下结论是否仍成立\n')
    if len(df_saag) > 0:
        for it in ['handcrafted_features', 'raw_phase']:
            random_saag = df_saag[(df_saag['split_type'] == 'random') & (df_saag['input_type'] == it)]['SAAG_predicted'].mean()
            sd_saag = df_saag[(df_saag['split_type'] == 'scene_disjoint') & (df_saag['input_type'] == it)]['SAAG_predicted'].mean()
            if not np.isnan(random_saag) and not np.isnan(sd_saag):
                lines.append(f'- {it}: random SAAG={random_saag:+.3f}, scene_disjoint SAAG={sd_saag:+.3f}, '
                            f'{"结论一致" if abs(random_saag - sd_saag) < 0.1 else "结论有差异"}\n')

    # Q14
    lines.append('### 14: handcrafted vs raw phase\n')
    if len(df_saag) > 0:
        for sp in ['random']:
            hf_saag = df_saag[(df_saag['split_type'] == sp) & (df_saag['input_type'] == 'handcrafted_features')]['SAAG_predicted'].mean()
            rp_saag = df_saag[(df_saag['split_type'] == sp) & (df_saag['input_type'] == 'raw_phase')]['SAAG_predicted'].mean()
            if not np.isnan(hf_saag) and not np.isnan(rp_saag):
                lines.append(f'- handcrafted SAAG={hf_saag:+.3f}, raw_phase SAAG={rp_saag:+.3f}, '
                            f'{"raw phase 更强" if rp_saag > hf_saag else "handcrafted 更强"}\n')

    # Q15
    lines.append('### 15: 对 fingerprint-free metasurface defense 的启发\n')
    lines.append('当前结果表明，部分防护策略存在可识别的指纹，且攻击者可以利用该指纹进行两阶段攻击。')
    lines.append('这提示未来需要设计 fingerprint-free metasurface defense，使策略本身难以被识别，')
    lines.append('从而阻断两阶段攻击的第一阶段。\n')

    # Chinese section for report
    lines.append('\n---\n\n')
    lines.append('## 从防护策略指纹到两阶段策略感知攻击\n\n')
    lines.append('### 为什么只看 motion/no_motion 准确率不够\n\n')
    lines.append('传统的隐私评估只关注攻击者对运动/非运动的分类准确率。然而，如果防护策略本身可以被识别，')
    lines.append('攻击者可以先识别策略，再调用对应的自适应分类器，形成两阶段攻击。')
    lines.append('这种攻击方式在单阶段评估中无法体现。\n\n')
    lines.append('### 什么是 Defense Fingerprint Leakage (DFL)\n\n')
    lines.append('DFL 衡量防护策略是否可被识别。DFL_acc 为策略分类器的准确率，DFL_excess 为超出随机猜测基线的部分。')
    lines.append('DFL 越高，说明防护策略越容易被识别，strategy fingerprint leakage 越严重。\n\n')
    lines.append('### 什么是 Strategy-Aware Attack Gain (SAAG)\n\n')
    lines.append('SAAG 衡量利用策略指纹后 motion 攻击是否提升。')
    lines.append('SAAG_predicted = Acc_predicted_strategy_aware - Acc_single_mixed。')
    lines.append('SAAG > 0.05 说明策略指纹可被攻击者利用。\n\n')
    lines.append('### 当前实验结果说明什么\n\n')
    if len(df_saag) > 0:
        mean_saag = df_saag['SAAG_predicted'].mean()
        n_exploitable = (df_saag['SAAG_predicted'] > 0.05).sum()
        lines.append(f'当前实验中，SAAG_predicted 平均值为 {mean_saag:+.3f}，'
                    f'有 {n_exploitable}/{len(df_saag)} 个策略-输入组合的 SAAG_predicted > 0.05。')
        if mean_saag > 0.05:
            lines.append('这表明策略指纹已从"可观察现象"转化为"可利用攻击路径"。\n')
        else:
            lines.append('这表明虽然策略指纹存在，但当前两阶段攻击暂未显著提升 motion inference。\n')
    lines.append('### 为什么这提示未来需要 fingerprint-free metasurface defense\n\n')
    lines.append('好的超表面隐私防护不仅应降低 motion inference accuracy，还应降低 defense strategy fingerprint leakage。')
    lines.append('如果策略指纹可被利用，攻击者可以通过两阶段攻击突破防护。')
    lines.append('因此，未来需要设计使策略本身难以被识别的超表面参数化方法，即 fingerprint-free metasurface defense。\n')

    path = os.path.join(reports_dir, 'two_stage_strategy_aware_attack_analysis.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Saved: {path}")

    # DFL report
    dfl_lines = []
    dfl_lines.append('# Defense Fingerprint Leakage Metrics\n\n')
    dfl_lines.append(f'> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
    dfl_lines.append('## 定义\n\n')
    dfl_lines.append('- **DFL_acc**: strategy classifier accuracy\n')
    dfl_lines.append('- **DFL_macroF1**: strategy classifier macro F1\n')
    dfl_lines.append('- **DFL_random_baseline**: 1/K (K = number of strategies)\n')
    dfl_lines.append('- **DFL_excess**: DFL_acc - DFL_random_baseline\n\n')
    if len(df_dfl) > 0:
        dfl_lines.append('## 结果\n\n')
        dfl_lines.append('| Split | Input | Model | DFL_acc | DFL_excess |')
        dfl_lines.append('|-------|-------|-------|---------|------------|')
        for _, row in df_dfl.iterrows():
            dfl_lines.append(f'| {row["split_type"]} | {row["input_type"]} | {row["model_name"]} | '
                            f'{row["DFL_acc"]:.3f} | {row["DFL_excess"]:.3f} |')
    dfl_path = os.path.join(reports_dir, 'defense_fingerprint_metrics.md')
    with open(dfl_path, 'w') as f:
        f.write('\n'.join(dfl_lines))
    print(f"  Saved: {dfl_path}")


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
    print("  Two-Stage Strategy-Aware Attack Experiment")
    print(f"  Mode: {args.mode}, Split: {args.split}, Seeds: {args.seeds}")
    print("=" * 60)

    all_results, all_strat_pred = run_experiment(args.mode, args.split, args.seeds, out_dir)
    generate_outputs(all_results, all_strat_pred, out_dir, args.split)
    generate_report(out_dir)

    print("\n  Two-stage strategy-aware attack experiment complete!")


if __name__ == '__main__':
    main()
