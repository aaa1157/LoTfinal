"""
Phase 1: 基础 RFID 隐私泄露与手工超表面仿真

回答 RQ1 和 RQ2:
- RQ1: 无超表面时，RFID 接收相位是否会泄露人体运动和呼吸隐私？
- RQ2: 1-bit 可编程超表面是否能扰乱 RFID 相位，从而降低传统攻击者的检测能力？

不依赖 torch

Usage:
    python main.py --mode debug
    python main.py --mode medium
    python main.py --mode full
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    RANDOM_SEED, fs as FS, T as T_VAL, sequence_length as LENGTH,
    strategies, default_respiration_freqs, noise_std_default,
)
from src.utils import make_rng, preprocess_phase
from src.signal_model import make_time_axis, simulate_received_signal
from src.metasurface import generate_metasurface_signal, compute_switching_rate
from src.dataset import build_dataset, load_dataset, filter_by_strategy, filter_by_task
from src.traditional_attacks import (
    variance_threshold_detection,
    statistical_classifier,
    respiration_estimation,
    periodic_filtering_analysis,
)
from src.metrics import (
    compute_phase_variance, compute_avg_phase_entropy,
    compute_avg_spectral_entropy, compute_avg_lowfreq_energy_ratio,
    compute_privacy_gain_accuracy, compute_privacy_gain_respiration,
)
from src.plots import (
    plot_phase_no_metasurface_motion_vs_nomotion,
    plot_phase_strategies_comparison,
    plot_spectrum_strategies_comparison,
    plot_walking_detection_metrics,
    plot_respiration_error_comparison,
    plot_summary_comparison,
    plot_multifreq_frequency_schedule,
    plot_dataset_distribution,
    plot_feature_pca,
    plot_confusion_matrices,
)


def ensure_dirs():
    """确保输出目录存在"""
    for d in ['results/figures', 'results/tables', 'results/models',
              'data/processed', 'data/splits', 'data/raw']:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)


def run_phase1(mode: str = "debug", split_type: str = "random", rebuild: bool = False):
    print("=" * 60)
    print("  Phase 1: RFID Privacy Leakage & Metasurface Defense")
    print("=" * 60)
    print(f"  Mode: {mode}, Split: {split_type}")

    ensure_dirs()

    # 1. 构建或加载数据集
    print("\n[1] Building/Loading dataset ...")
    data = build_dataset(mode=mode, split_type=split_type, force=rebuild,
                         seed=RANDOM_SEED, base_dir=PROJECT_ROOT)

    metadata_df = data['metadata_df']
    X_phase = data['X_phase']
    y_motion = data['y_motion']
    strategy_index = data['strategy_index']
    task_index = data['task_index']
    _mode = mode
    _split = split_type

    # 2. 数据集分布图
    print("\n[2] Plotting dataset distribution ...")
    plot_dataset_distribution(metadata_df,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/dataset_distribution.png'))

    # 3. PCA 特征分布图
    print("\n[3] Plotting PCA features ...")
    walk_mask = task_index == 0
    plot_feature_pca(
        X_phase[walk_mask], y_motion[walk_mask],
        strategy_index[walk_mask], strategies,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/feature_tsne_or_pca.png')
    )

    # 4. 生成示例信号用于绘图
    print("\n[4] Generating sample signals for visualization ...")
    t = make_time_axis()
    rng = make_rng(RANDOM_SEED)

    # no_metasurface 示例
    sample_phases = {}
    sample_motion = {}
    sample_nomotion = {}

    for strategy_name in strategies:
        meta_rng = make_rng(rng.integers(0, 2**31))
        v_meta, states, sched_info = generate_metasurface_signal(strategy_name, t, rng=meta_rng)

        sig_rng = make_rng(rng.integers(0, 2**31))
        _, phi_motion, _ = simulate_received_signal(
            t, strategy_name=strategy_name, motion_label=1,
            task_type="walking_detection", rng=sig_rng,
            meta_component=v_meta,
        )

        meta_rng2 = make_rng(rng.integers(0, 2**31))
        v_meta2, _, _ = generate_metasurface_signal(strategy_name, t, rng=meta_rng2)
        sig_rng2 = make_rng(rng.integers(0, 2**31))
        _, phi_nomotion, _ = simulate_received_signal(
            t, strategy_name=strategy_name, motion_label=0,
            task_type="walking_detection", rng=sig_rng2,
            meta_component=v_meta2,
        )

        sample_phases[strategy_name] = phi_motion
        sample_motion[strategy_name] = phi_motion
        sample_nomotion[strategy_name] = phi_nomotion

        # 保存 multifreq schedule
        if strategy_name == 'multifreq_proposed' and sched_info:
            schedule_rows = []
            for entry in sched_info:
                schedule_rows.append({
                    'time': entry['time'],
                    'subarray_id': entry['subarray_id'],
                    'new_freq': entry['new_freq'],
                })
            if schedule_rows:
                pd.DataFrame(schedule_rows).to_csv(
                    os.path.join(PROJECT_ROOT, 'results/tables/multifreq_schedule.csv'),
                    index=False
                )
            plot_multifreq_frequency_schedule(sched_info,
                save_path=os.path.join(PROJECT_ROOT, 'results/figures/multifreq_frequency_schedule.png'))

    # 5. 绘制基础图
    print("\n[5] Plotting basic figures ...")
    plot_phase_no_metasurface_motion_vs_nomotion(
        sample_motion['no_metasurface'], sample_nomotion['no_metasurface'], FS,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/phase_no_metasurface_motion_vs_nomotion.png'))

    plot_phase_strategies_comparison(
        sample_phases, FS,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/phase_strategies_comparison.png'))

    plot_spectrum_strategies_comparison(
        sample_phases, FS,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/spectrum_strategies_comparison.png'))

    # 6. 传统攻击评估
    print("\n[6] Running traditional attacks ...")
    all_metrics = {}
    all_resp_results = {}
    confusion_matrices = {}

    for strategy_name in strategies:
        print(f"\n  === Strategy: {strategy_name} ===")

        # 筛选该策略的 walking 数据
        walk_data = filter_by_strategy(
            filter_by_task(data, 'walking_detection'), strategy_name
        )

        motion_phases = [walk_data['X_phase'][i] for i in range(len(walk_data['X_phase']))
                         if walk_data['y_motion'][i] == 1]
        nomotion_phases = [walk_data['X_phase'][i] for i in range(len(walk_data['X_phase']))
                           if walk_data['y_motion'][i] == 0]

        # 攻击1: 方差阈值
        thresh_result = variance_threshold_detection(motion_phases, nomotion_phases)
        print(f"    Threshold: acc={thresh_result['accuracy']:.3f}, "
              f"TPR={thresh_result['tpr']:.3f}, FPR={thresh_result['fpr']:.3f}")

        # 攻击2: 统计分类器
        clf_result = statistical_classifier(motion_phases, nomotion_phases, FS)
        print(f"    Classifier: acc={clf_result['accuracy']:.3f}, "
              f"TPR={clf_result['tpr']:.3f}, FPR={clf_result['fpr']:.3f}")

        # 混淆矩阵
        if 'confusion_matrix' in clf_result:
            confusion_matrices[strategy_name] = clf_result['confusion_matrix']

        # 相位指标
        all_phases = motion_phases + nomotion_phases
        phase_var = compute_phase_variance(all_phases)
        phase_ent = compute_avg_phase_entropy(all_phases)
        spec_ent = compute_avg_spectral_entropy(all_phases, FS)
        lf_ratio = compute_avg_lowfreq_energy_ratio(all_phases, FS)

        # 切换率
        meta_rng3 = make_rng(rng.integers(0, 2**31))
        _, states_sample, _ = generate_metasurface_signal(strategy_name, t, rng=meta_rng3)
        sw_rate = compute_switching_rate(states_sample, FS)

        all_metrics[strategy_name] = {
            'threshold_accuracy': thresh_result['accuracy'],
            'threshold_tpr': thresh_result['tpr'],
            'threshold_fpr': thresh_result['fpr'],
            'classifier_accuracy': clf_result['accuracy'],
            'classifier_tpr': clf_result['tpr'],
            'classifier_fpr': clf_result['fpr'],
            'precision': clf_result.get('precision', 0.0),
            'recall': clf_result.get('recall', 0.0),
            'f1': clf_result.get('f1', 0.0),
            'mean_phase_variance': phase_var,
            'mean_phase_entropy': phase_ent,
            'mean_spectral_entropy': spec_ent,
            'mean_lowfreq_energy_ratio': lf_ratio,
            'switching_rate': sw_rate,
        }

        # 呼吸估计
        resp_data = filter_by_strategy(
            filter_by_task(data, 'respiration_estimation'), strategy_name
        )
        resp_phases = [resp_data['X_phase'][i] for i in range(len(resp_data['X_phase']))]
        # 获取真实呼吸频率
        resp_meta = resp_data['metadata_df']
        true_freqs = resp_meta['true_respiration_freq'].dropna().values
        true_freq = float(true_freqs[0]) if len(true_freqs) > 0 else 0.25

        resp_result = respiration_estimation(resp_phases, true_freq, FS)
        all_resp_results[strategy_name] = resp_result
        print(f"    Respiration: error={resp_result['mean_abs_error']:.4f} Hz")

    # 7. 计算隐私增益
    baseline_acc = all_metrics['no_metasurface']['classifier_accuracy']
    baseline_resp_error = all_resp_results['no_metasurface']['mean_abs_error']

    for strategy_name in strategies:
        pg_acc = compute_privacy_gain_accuracy(
            baseline_acc, all_metrics[strategy_name]['classifier_accuracy'])
        pg_resp = compute_privacy_gain_respiration(
            baseline_resp_error, all_resp_results[strategy_name]['mean_abs_error'])
        all_metrics[strategy_name]['privacy_gain_accuracy'] = pg_acc
        all_resp_results[strategy_name]['privacy_gain_respiration_error'] = pg_resp

    # 8. 保存结果表格
    print("\n[7] Saving result tables ...")

    # metrics.csv
    metrics_rows = []
    for strategy_name in strategies:
        row = {'strategy': strategy_name}
        row.update(all_metrics[strategy_name])
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(
        os.path.join(PROJECT_ROOT, 'results/tables/metrics.csv'), index=False)
    print("  Saved: results/tables/metrics.csv")

    # respiration_errors.csv
    resp_rows = []
    for strategy_name in strategies:
        row = {'strategy': strategy_name}
        row.update(all_resp_results[strategy_name])
        resp_rows.append(row)
    pd.DataFrame(resp_rows).to_csv(
        os.path.join(PROJECT_ROOT, 'results/tables/respiration_errors.csv'), index=False)
    print("  Saved: results/tables/respiration_errors.csv")

    # dataset_summary.csv
    summary = metadata_df.groupby(['strategy', 'motion_label', 'task_type', 'split']).size().reset_index(name='count')
    summary.to_csv(os.path.join(PROJECT_ROOT, 'results/tables/dataset_summary.csv'), index=False)
    print("  Saved: results/tables/dataset_summary.csv")

    # feature_importance
    if 'feature_coefficients' in all_metrics.get('no_metasurface', {}):
        coef_dict = all_metrics['no_metasurface']['feature_coefficients']
        pd.DataFrame([coef_dict]).to_csv(
            os.path.join(PROJECT_ROOT, 'results/tables/feature_importance_or_coefficients.csv'), index=False)
        print("  Saved: results/tables/feature_importance_or_coefficients.csv")

    # 9. 绘制结果图
    print("\n[8] Plotting result figures ...")
    plot_walking_detection_metrics(all_metrics,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/walking_detection_metrics.png'))

    plot_respiration_error_comparison(all_resp_results,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/respiration_error_comparison.png'))

    plot_summary_comparison(all_metrics,
        save_path=os.path.join(PROJECT_ROOT, 'results/figures/summary_comparison.png'))

    if confusion_matrices:
        plot_confusion_matrices(confusion_matrices,
            save_path=os.path.join(PROJECT_ROOT, 'results/figures/confusion_matrices.png'))

    # 10. 打印结果摘要
    print("\n" + "=" * 60)
    print("  Phase 1 Results Summary")
    print("=" * 60)
    print(f"\n{'Strategy':<22} {'Thresh Acc':>10} {'Clf Acc':>10} {'Clf TPR':>10} "
          f"{'Clf FPR':>10} {'Phase Ent':>10} {'LF Ratio':>10} {'Sw Rate':>10} "
          f"{'Resp Err':>10} {'PG Acc':>10}")
    print("-" * 122)

    for strategy_name in strategies:
        m = all_metrics[strategy_name]
        r = all_resp_results[strategy_name]
        print(f"{strategy_name:<22} {m['threshold_accuracy']:>10.3f} {m['classifier_accuracy']:>10.3f} "
              f"{m['classifier_tpr']:>10.3f} {m['classifier_fpr']:>10.3f} "
              f"{m['mean_phase_entropy']:>10.3f} {m['mean_lowfreq_energy_ratio']:>10.3f} "
              f"{m['switching_rate']:>10.3f} {r['mean_abs_error']:>10.4f} "
              f"{m['privacy_gain_accuracy']:>10.3f}")

    print("\n  Phase 1 complete!")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Basic RFID Privacy Simulation")
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['debug', 'medium', 'full'],
                        help='Experiment mode')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'scene_disjoint'],
                        help='Split type')
    parser.add_argument('--fast', action='store_true', help='Alias for --mode debug')
    parser.add_argument('--rebuild-dataset', action='store_true', help='Force rebuild dataset')
    args = parser.parse_args()

    mode = 'debug' if args.fast else args.mode
    run_phase1(mode=mode, split_type=args.split, rebuild=args.rebuild_dataset)


if __name__ == '__main__':
    main()
