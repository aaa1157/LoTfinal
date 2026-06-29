"""
Strategy fingerprint deep-dive experiment.

This CPU-only script tests whether a metasurface defense strategy leaves a
recognizable fingerprint in RFID phase features, and whether an attacker can
use the fingerprint for a two-stage motion attack.

Run from the project directory:
    python scripts/run_strategy_fingerprint_deepdive_v2.py --mode medium

Outputs are written to:
    results/strategy_fingerprint_deepdive_v2/
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import fs as FS
from src.config import strategies
from src.dataset import load_dataset
from src.utils import FEATURE_NAMES, extract_features


STRATEGY_LIST = list(strategies)
FEATURE_COLUMNS = list(FEATURE_NAMES)


@dataclass(frozen=True)
class SplitArrays:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    s_train: np.ndarray
    s_val: np.ndarray
    s_test: np.ndarray
    test_meta: pd.DataFrame


def make_strategy_classifiers(seed: int) -> Dict[str, Pipeline]:
    return {
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        C=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "RF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)),
            ]
        ),
        "GBDT": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=160,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def make_motion_classifiers(seed: int) -> Dict[str, Pipeline]:
    return {
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=seed)),
            ]
        ),
        "RF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=220,
                        max_depth=12,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)),
            ]
        ),
        "GBDT": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=160,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def extract_feature_matrix(phases: np.ndarray) -> np.ndarray:
    features = np.asarray([extract_features(phase, FS) for phase in phases], dtype=np.float64)
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


def load_walking_split(mode: str, split_type: str, seed: int, base_dir: Path) -> SplitArrays:
    data = load_dataset(mode=mode, split_type=split_type, base_dir=str(base_dir))

    walking_mask = data["task_index"] == 0
    phases = data["X_phase"][walking_mask]
    y_motion = data["y_motion"][walking_mask].astype(int)
    strategy_index = data["strategy_index"][walking_mask].astype(int)
    meta = data["metadata_df"][walking_mask].reset_index(drop=True)

    x_features = extract_feature_matrix(phases)

    if split_type == "scene_disjoint":
        train_val_mask = meta["scene_id"].isin(range(0, 20)).to_numpy()
        test_mask = meta["scene_id"].isin(range(20, 30)).to_numpy()
        train_val_indices = np.where(train_val_mask)[0]
        test_indices = np.where(test_mask)[0]
        rng = np.random.default_rng(seed + 101)
        train_val_indices = rng.permutation(train_val_indices)
        n_train = int(len(train_val_indices) * 0.824)
        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]

        train_scenes = set(meta.loc[train_indices, "scene_id"].astype(int))
        test_scenes = set(meta.loc[test_indices, "scene_id"].astype(int))
        overlap = train_scenes & test_scenes
        if overlap:
            raise RuntimeError(f"scene_disjoint split has scene overlap: {sorted(overlap)}")
    else:
        rng = np.random.default_rng(seed + 101)
        indices = rng.permutation(np.arange(len(y_motion)))
        n_train = int(len(indices) * 0.70)
        n_val = int(len(indices) * 0.15)
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

    return SplitArrays(
        x_train=x_features[train_indices],
        x_val=x_features[val_indices],
        x_test=x_features[test_indices],
        y_train=y_motion[train_indices],
        y_val=y_motion[val_indices],
        y_test=y_motion[test_indices],
        s_train=strategy_index[train_indices],
        s_val=strategy_index[val_indices],
        s_test=strategy_index[test_indices],
        test_meta=meta.loc[test_indices].reset_index(drop=True),
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["auc"] = np.nan
    return metrics


def predict_with_score(model: Pipeline, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray | None]:
    y_pred = model.predict(x).astype(int)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(x)
    else:
        score = None
    return y_pred, score


def fit_best_strategy_classifier(arrays: SplitArrays, seed: int) -> Tuple[str, Pipeline, Dict[str, float]]:
    rows = []
    best_name = ""
    best_model = None
    best_key = (-1.0, -1.0)

    for name, template in make_strategy_classifiers(seed).items():
        model = clone(template)
        model.fit(arrays.x_train, arrays.s_train)
        val_pred = model.predict(arrays.x_val)
        val_acc = accuracy_score(arrays.s_val, val_pred)
        val_macro_f1 = f1_score(arrays.s_val, val_pred, average="macro", zero_division=0)
        rows.append({"strategy_classifier": name, "val_accuracy": val_acc, "val_macro_f1": val_macro_f1})
        key = (val_acc, val_macro_f1)
        if key > best_key:
            best_key = key
            best_name = name
            best_model = model

    assert best_model is not None
    strategy_pred = best_model.predict(arrays.x_test)
    metrics = {
        "strategy_classifier": best_name,
        "strategy_val_accuracy": best_key[0],
        "strategy_val_macro_f1": best_key[1],
        "strategy_test_accuracy": accuracy_score(arrays.s_test, strategy_pred),
        "strategy_test_macro_f1": f1_score(arrays.s_test, strategy_pred, average="macro", zero_division=0),
    }
    return best_name, best_model, metrics


def train_strategy_specific_models(
    arrays: SplitArrays,
    classifier_name: str,
    seed: int,
) -> Dict[int, Pipeline]:
    template = make_motion_classifiers(seed)[classifier_name]
    models = {}
    for strategy_id in range(len(STRATEGY_LIST)):
        mask = arrays.s_train == strategy_id
        if mask.sum() < 20:
            continue
        model = clone(template)
        model.fit(arrays.x_train[mask], arrays.y_train[mask])
        models[strategy_id] = model
    return models


def predict_strategy_aware(
    models: Dict[int, Pipeline],
    x_test: np.ndarray,
    selected_strategy: np.ndarray,
    fallback: Pipeline,
) -> Tuple[np.ndarray, np.ndarray | None]:
    y_pred = np.zeros(len(x_test), dtype=int)
    y_score = np.zeros(len(x_test), dtype=float)
    has_score = False

    for strategy_id in np.unique(selected_strategy):
        mask = selected_strategy == strategy_id
        model = models.get(int(strategy_id), fallback)
        pred, score = predict_with_score(model, x_test[mask])
        y_pred[mask] = pred
        if score is not None:
            y_score[mask] = score
            has_score = True

    return y_pred, y_score if has_score else None


def append_strategy_rows(
    rows: List[Dict[str, object]],
    *,
    mode: str,
    split_type: str,
    seed: int,
    attack_type: str,
    motion_classifier: str,
    strategy_classifier: str,
    strategy_metrics: Dict[str, float],
    arrays: SplitArrays,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    selected_strategy: np.ndarray | None,
) -> None:
    overall = binary_metrics(arrays.y_test, y_pred, y_score)
    rows.append(
        {
            "mode": mode,
            "split_type": split_type,
            "seed": seed,
            "attack_type": attack_type,
            "test_strategy": "ALL",
            "motion_classifier": motion_classifier,
            "strategy_classifier": strategy_classifier,
            "strategy_selection_accuracy": np.nan
            if selected_strategy is None
            else accuracy_score(arrays.s_test, selected_strategy),
            "num_test": len(arrays.y_test),
            **strategy_metrics,
            **overall,
        }
    )

    for strategy_id, strategy_name in enumerate(STRATEGY_LIST):
        mask = arrays.s_test == strategy_id
        if mask.sum() == 0:
            continue
        score_subset = None if y_score is None else y_score[mask]
        metrics = binary_metrics(arrays.y_test[mask], y_pred[mask], score_subset)
        rows.append(
            {
                "mode": mode,
                "split_type": split_type,
                "seed": seed,
                "attack_type": attack_type,
                "test_strategy": strategy_name,
                "motion_classifier": motion_classifier,
                "strategy_classifier": strategy_classifier,
                "strategy_selection_accuracy": np.nan
                if selected_strategy is None
                else accuracy_score(arrays.s_test[mask], selected_strategy[mask]),
                "num_test": int(mask.sum()),
                **strategy_metrics,
                **metrics,
            }
        )


def run_one_setting(mode: str, split_type: str, seed: int, base_dir: Path) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    arrays = load_walking_split(mode, split_type, seed, base_dir)
    _, strategy_model, strategy_metrics = fit_best_strategy_classifier(arrays, seed)
    strategy_pred = strategy_model.predict(arrays.x_test).astype(int)
    cm = confusion_matrix(arrays.s_test, strategy_pred, labels=np.arange(len(STRATEGY_LIST)))
    cm_df = pd.DataFrame(cm, index=STRATEGY_LIST, columns=STRATEGY_LIST)
    cm_df.index.name = "true_strategy"
    cm_df.columns.name = "predicted_strategy"

    rows: List[Dict[str, object]] = []
    empty_strategy_metrics = {
        "strategy_val_accuracy": np.nan,
        "strategy_val_macro_f1": np.nan,
        "strategy_test_accuracy": np.nan,
        "strategy_test_macro_f1": np.nan,
    }

    for motion_name, template in make_motion_classifiers(seed).items():
        single_model = clone(template)
        single_model.fit(arrays.x_train, arrays.y_train)
        single_pred, single_score = predict_with_score(single_model, arrays.x_test)
        append_strategy_rows(
            rows,
            mode=mode,
            split_type=split_type,
            seed=seed,
            attack_type="single_mixed",
            motion_classifier=motion_name,
            strategy_classifier="none",
            strategy_metrics=empty_strategy_metrics,
            arrays=arrays,
            y_pred=single_pred,
            y_score=single_score,
            selected_strategy=None,
        )

        strategy_models = train_strategy_specific_models(arrays, motion_name, seed)
        oracle_pred, oracle_score = predict_strategy_aware(
            strategy_models, arrays.x_test, arrays.s_test, fallback=single_model
        )
        append_strategy_rows(
            rows,
            mode=mode,
            split_type=split_type,
            seed=seed,
            attack_type="oracle_strategy_aware",
            motion_classifier=motion_name,
            strategy_classifier="oracle",
            strategy_metrics={**strategy_metrics, "strategy_classifier": "oracle"},
            arrays=arrays,
            y_pred=oracle_pred,
            y_score=oracle_score,
            selected_strategy=arrays.s_test,
        )

        predicted_pred, predicted_score = predict_strategy_aware(
            strategy_models, arrays.x_test, strategy_pred, fallback=single_model
        )
        append_strategy_rows(
            rows,
            mode=mode,
            split_type=split_type,
            seed=seed,
            attack_type="predicted_strategy_aware",
            motion_classifier=motion_name,
            strategy_classifier=strategy_metrics["strategy_classifier"],
            strategy_metrics=strategy_metrics,
            arrays=arrays,
            y_pred=predicted_pred,
            y_score=predicted_score,
            selected_strategy=strategy_pred,
        )

    return rows, cm_df


def summarize_gains(result_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["mode", "split_type", "seed", "test_strategy", "motion_classifier"]
    pivot = result_df.pivot_table(
        index=key_cols,
        columns="attack_type",
        values=["accuracy", "balanced_accuracy", "f1", "auc"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{attack}" for metric, attack in pivot.columns]
    pivot = pivot.reset_index()

    for metric in ["accuracy", "balanced_accuracy", "f1", "auc"]:
        base = f"{metric}_single_mixed"
        pred = f"{metric}_predicted_strategy_aware"
        oracle = f"{metric}_oracle_strategy_aware"
        if base in pivot.columns and pred in pivot.columns:
            pivot[f"{metric}_predicted_gain"] = pivot[pred] - pivot[base]
        if base in pivot.columns and oracle in pivot.columns:
            pivot[f"{metric}_oracle_gain"] = pivot[oracle] - pivot[base]
    return pivot


def write_report(
    out_dir: Path,
    mode: str,
    splits: Iterable[str],
    seeds: Iterable[int],
    result_df: pd.DataFrame,
    gain_df: pd.DataFrame,
) -> None:
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_rows = result_df[result_df["test_strategy"] == "ALL"]
    all_summary = (
        all_rows.groupby(["split_type", "attack_type"])[
            ["accuracy", "balanced_accuracy", "f1", "auc", "strategy_test_accuracy", "strategy_test_macro_f1"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    gain_summary = (
        gain_df[gain_df["test_strategy"] == "ALL"]
        .groupby(["split_type"])[
            [
                "accuracy_predicted_gain",
                "balanced_accuracy_predicted_gain",
                "f1_predicted_gain",
                "accuracy_oracle_gain",
                "balanced_accuracy_oracle_gain",
                "f1_oracle_gain",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )

    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        render = df.copy()
        for col in render.columns:
            if pd.api.types.is_float_dtype(render[col]):
                render[col] = render[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
            else:
                render[col] = render[col].astype(str)
        header = "| " + " | ".join(render.columns) + " |"
        divider = "| " + " | ".join(["---"] * len(render.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in render.to_numpy(dtype=str)]
        return "\n".join([header, divider, *body])

    lines = [
        "# Strategy Fingerprint Deep-Dive V2",
        "",
        f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- mode: {mode}",
        f"- splits: {', '.join(splits)}",
        f"- seeds: {', '.join(str(s) for s in seeds)}",
        f"- strategies: {', '.join(STRATEGY_LIST)}",
        f"- feature_count: {len(FEATURE_COLUMNS)}",
        "",
        "## Experiment",
        "",
        "The experiment uses walking-detection samples only. Stage 1 trains a strategy classifier from 12 statistical phase features. Stage 2 trains strategy-specific motion classifiers. The predicted strategy-aware attack selects the motion classifier using the Stage 1 prediction.",
        "",
        "## Overall Results",
        "",
        markdown_table(all_summary),
        "",
        "## Attack Gain Over Single Mixed Attacker",
        "",
        markdown_table(gain_summary),
        "",
        "## Output Files",
        "",
        "- tables/two_stage_results.csv",
        "- tables/two_stage_gain.csv",
        "- tables/overall_summary.csv",
        "- confusion_matrices/*.csv",
    ]

    (report_dir / "strategy_fingerprint_deepdive_v2_report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(out_dir: Path, gain_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    subset = gain_df[gain_df["test_strategy"] == "ALL"]
    summary = (
        subset.groupby(["split_type"])[
            ["accuracy_predicted_gain", "accuracy_oracle_gain", "f1_predicted_gain", "f1_oracle_gain"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )

    x = np.arange(len(summary))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    for offset, col, label in [
        (-1.5, "accuracy_predicted_gain", "Predicted Acc Gain"),
        (-0.5, "accuracy_oracle_gain", "Oracle Acc Gain"),
        (0.5, "f1_predicted_gain", "Predicted F1 Gain"),
        (1.5, "f1_oracle_gain", "Oracle F1 Gain"),
    ]:
        ax.bar(x + offset * width, summary[col], width=width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["split_type"])
    ax.set_ylabel("Gain over single mixed attacker")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "strategy_aware_attack_gain.png")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU strategy-fingerprint deep-dive experiment")
    parser.add_argument("--mode", default="medium", choices=["debug", "medium", "full"])
    parser.add_argument("--splits", nargs="+", default=["random", "scene_disjoint"], choices=["random", "scene_disjoint"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--base-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "strategy_fingerprint_deepdive_v2"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    table_dir = out_dir / "tables"
    cm_dir = out_dir / "confusion_matrices"
    table_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    print(f"[INFO] output: {out_dir}")
    for split_type in args.splits:
        for seed in args.seeds:
            print(f"[INFO] running mode={args.mode} split={split_type} seed={seed}")
            rows, cm_df = run_one_setting(args.mode, split_type, seed, base_dir)
            all_rows.extend(rows)
            cm_path = cm_dir / f"strategy_confusion_{args.mode}_{split_type}_seed{seed}.csv"
            cm_df.to_csv(cm_path, encoding="utf-8-sig")

    result_df = pd.DataFrame(all_rows)
    result_path = table_dir / "two_stage_results.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    gain_df = summarize_gains(result_df)
    gain_path = table_dir / "two_stage_gain.csv"
    gain_df.to_csv(gain_path, index=False, encoding="utf-8-sig")

    overall_summary = (
        result_df.groupby(["mode", "split_type", "attack_type", "motion_classifier", "test_strategy"])
        [["accuracy", "balanced_accuracy", "f1", "auc", "strategy_selection_accuracy"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    overall_summary.to_csv(table_dir / "overall_summary.csv", index=False, encoding="utf-8-sig")

    write_report(out_dir, args.mode, args.splits, args.seeds, result_df, gain_df)
    plot_summary(out_dir, gain_df)

    print("[DONE] result files:")
    print(f"  {result_path}")
    print(f"  {gain_path}")
    print(f"  {table_dir / 'overall_summary.csv'}")
    print(f"  {out_dir / 'reports' / 'strategy_fingerprint_deepdive_v2_report.md'}")


if __name__ == "__main__":
    main()
