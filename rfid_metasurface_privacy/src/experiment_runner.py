"""
实验运行器 - 统一管理所有实验的运行

不依赖 torch
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Optional

from src.config import strategies, RANDOM_SEED, fs as FS
from src.dataset import build_dataset, load_dataset, filter_by_strategy, filter_by_task
from src.traditional_attacks import (
    variance_threshold_detection,
    statistical_classifier,
    respiration_estimation,
)
from src.metrics import (
    compute_phase_variance, compute_avg_phase_entropy,
    compute_avg_spectral_entropy, compute_avg_lowfreq_energy_ratio,
    compute_privacy_gain_accuracy, compute_privacy_gain_respiration,
)
from src.signal_model import make_time_axis, simulate_received_signal
from src.metasurface import generate_metasurface_signal, compute_switching_rate
from src.utils import make_rng


def run_traditional_attacks(data: Dict, base_dir: str) -> Dict:
    """
    运行所有传统攻击实验

    Returns:
        dict with metrics and resp_results for each strategy
    """
    t = make_time_axis()
    rng = make_rng(RANDOM_SEED + 300)

    all_metrics = {}
    all_resp_results = {}

    for strategy_name in strategies:
        walk_data = filter_by_strategy(
            filter_by_task(data, 'walking_detection'), strategy_name
        )

        motion_phases = [walk_data['X_phase'][i] for i in range(len(walk_data['X_phase']))
                         if walk_data['y_motion'][i] == 1]
        nomotion_phases = [walk_data['X_phase'][i] for i in range(len(walk_data['X_phase']))
                           if walk_data['y_motion'][i] == 0]

        thresh_result = variance_threshold_detection(motion_phases, nomotion_phases)
        clf_result = statistical_classifier(motion_phases, nomotion_phases, FS)

        all_phases = motion_phases + nomotion_phases
        phase_var = compute_phase_variance(all_phases)
        phase_ent = compute_avg_phase_entropy(all_phases)
        spec_ent = compute_avg_spectral_entropy(all_phases, FS)
        lf_ratio = compute_avg_lowfreq_energy_ratio(all_phases, FS)

        meta_rng = make_rng(rng.integers(0, 2**31))
        _, states_sample, _ = generate_metasurface_signal(strategy_name, t, rng=meta_rng)
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

        # Respiration
        resp_data = filter_by_strategy(
            filter_by_task(data, 'respiration_estimation'), strategy_name
        )
        resp_phases = [resp_data['X_phase'][i] for i in range(len(resp_data['X_phase']))]
        resp_meta = resp_data['metadata_df']
        true_freqs = resp_meta['true_respiration_freq'].dropna().values
        true_freq = float(true_freqs[0]) if len(true_freqs) > 0 else 0.25
        resp_result = respiration_estimation(resp_phases, true_freq, FS)
        all_resp_results[strategy_name] = resp_result

    # Privacy gain
    baseline_acc = all_metrics['no_metasurface']['classifier_accuracy']
    baseline_resp_error = all_resp_results['no_metasurface']['mean_abs_error']

    for strategy_name in strategies:
        all_metrics[strategy_name]['privacy_gain_accuracy'] = compute_privacy_gain_accuracy(
            baseline_acc, all_metrics[strategy_name]['classifier_accuracy'])
        all_resp_results[strategy_name]['privacy_gain_respiration_error'] = compute_privacy_gain_respiration(
            baseline_resp_error, all_resp_results[strategy_name]['mean_abs_error'])

    return {'metrics': all_metrics, 'resp_results': all_resp_results}
