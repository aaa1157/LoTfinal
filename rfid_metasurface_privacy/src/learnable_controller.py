"""
可学习超表面控制器 (LMC: Learnable Metasurface Controller)

黑盒优化控制器：输出 multifreq_proposed 的控制参数
通过随机搜索优化参数

依赖 torch（用于冻结攻击者模型评估）
"""

import os
import numpy as np
from typing import Dict, Optional, List, Tuple

from src.config import (
    RANDOM_SEED, fs as FS, T as T_VAL, sequence_length as LENGTH,
    human_motion_band, lmc_modes, lmc_alpha, lmc_beta, lmc_eta, lmc_gamma,
)
from src.utils import (
    make_rng, preprocess_phase, compute_phase_entropy, compute_spectral_entropy,
    compute_lowfreq_energy_ratio,
)
from src.signal_model import make_time_axis, simulate_received_signal
from src.metasurface import generate_metasurface_signal, compute_switching_rate


class LMCController:
    """
    Learnable Metasurface Controller

    黑盒优化：随机搜索 multifreq_proposed 的控制参数
    优化目标：最小化 objective = attacker_accuracy
              - alpha * respiration_error_normalized
              - beta * spectral_entropy_normalized
              - eta * lowfreq_energy_ratio
              + gamma * switching_rate_normalized
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.rng = make_rng(seed)
        self.best_params = None
        self.best_objective = float('inf')
        self.best_attacker_acc = 1.0
        self.search_history = []

    def sample_params(self) -> Dict:
        """随机采样一组控制参数"""
        return {
            'init_freqs': list(self.rng.uniform(0.2, 2.5, size=4)),
            'random_flip_prob': float(self.rng.uniform(0.05, 0.30)),
            'regroup_prob': float(self.rng.uniform(0.0, 0.05)),
            'small_ratio': float(self.rng.uniform(0.15, 0.40)),
            'large_ratio': float(self.rng.uniform(0.55, 0.90)),
            'lowfreq_weight': float(self.rng.uniform(0.4, 0.85)),
            'hop_interval_range': (
                float(self.rng.uniform(2, 5)),
                float(self.rng.uniform(6, 12)),
            ),
            'metasurface_amplitude': float(self.rng.uniform(0.05, 0.15)),
            'switching_penalty': float(self.rng.uniform(0.0, 0.3)),
        }

    def evaluate_params(
        self,
        params: Dict,
        attacker_model,
        device,
        n_eval_samples: int = 50,
        n_seeds: int = 1,
    ) -> Dict:
        """
        评估一组控制参数（多 seed 评估取平均）
        """
        import torch

        t = make_time_axis()

        all_attacker_accs = []
        all_phase_entropies = []
        all_spectral_entropies = []
        all_lf_ratios = []
        all_sw_rates = []
        all_resp_errors = []

        for seed_idx in range(n_seeds):
            eval_rng = make_rng(self.rng.integers(0, 2**31) + seed_idx)
            correct_count = 0
            total_count = 0
            phase_entropies = []
            spectral_entropies = []
            lf_ratios = []
            sw_rates = []

            for i in range(n_eval_samples):
                sample_rng = make_rng(eval_rng.integers(0, 2**31))
                motion_label = int(sample_rng.integers(0, 2))

                meta_rng = make_rng(sample_rng.integers(0, 2**31))
                v_meta, states, _ = generate_metasurface_signal(
                    'multifreq_proposed', t, rng=meta_rng, params=params
                )

                _, phi, _ = simulate_received_signal(
                    t, strategy_name='lmc_controller',
                    motion_label=motion_label,
                    task_type="walking_detection",
                    rng=sample_rng,
                    meta_component=v_meta,
                )

                p = preprocess_phase(phi, detrend=True, zero_mean=True, standardize=True)
                x = torch.FloatTensor(p).unsqueeze(0).unsqueeze(0).to(device)

                attacker_model.eval()
                with torch.no_grad():
                    out = attacker_model(x)
                    pred = out.argmax(dim=1).item()

                if pred == motion_label:
                    correct_count += 1
                total_count += 1

                phase_entropies.append(compute_phase_entropy(p))
                spectral_entropies.append(compute_spectral_entropy(p, FS))
                lf_ratios.append(compute_lowfreq_energy_ratio(p, FS))
                sw_rates.append(compute_switching_rate(states, FS))

            attacker_acc = correct_count / total_count if total_count > 0 else 0.0
            all_attacker_accs.append(attacker_acc)
            all_phase_entropies.append(np.mean(phase_entropies))
            all_spectral_entropies.append(np.mean(spectral_entropies))
            all_lf_ratios.append(np.mean(lf_ratios))
            all_sw_rates.append(np.mean(sw_rates))

        mean_attacker_acc = np.mean(all_attacker_accs)
        mean_phase_entropy = np.mean(all_phase_entropies)
        mean_spectral_entropy = np.mean(all_spectral_entropies)
        mean_lf_ratio = np.mean(all_lf_ratios)
        mean_sw_rate = np.mean(all_sw_rates)

        # 归一化
        entropy_norm = mean_spectral_entropy / 10.0
        lf_norm = mean_lf_ratio
        sw_norm = mean_sw_rate / 10.0
        resp_error_norm = 0.5  # 简化：无呼吸评估时用默认值

        objective = (mean_attacker_acc
                     - lmc_alpha * resp_error_norm
                     - lmc_beta * entropy_norm
                     - lmc_eta * lf_norm
                     + lmc_gamma * sw_norm)

        return {
            'attacker_accuracy': mean_attacker_acc,
            'phase_entropy': mean_phase_entropy,
            'spectral_entropy': mean_spectral_entropy,
            'lowfreq_energy_ratio': mean_lf_ratio,
            'switching_rate': mean_sw_rate,
            'objective': objective,
        }

    def search(
        self,
        attacker_model,
        device,
        mode: str = "debug",
    ) -> Dict:
        """
        随机搜索最优控制参数

        Returns:
            best_params, best_metrics, search_history
        """
        mode_config = lmc_modes.get(mode, lmc_modes["debug"])
        n_iters = mode_config["search_iters"]
        n_eval_seeds = mode_config["eval_seeds"]
        n_eval_samples = 30 if mode == "debug" else 50

        self.best_params = self.sample_params()
        self.best_objective = float('inf')
        self.best_attacker_acc = 1.0
        self.search_history = []

        print(f"  LMC Search: {n_iters} iters, {n_eval_seeds} eval seeds, {n_eval_samples} samples each")

        for i in range(n_iters):
            params = self.sample_params()
            metrics = self.evaluate_params(
                params, attacker_model, device, n_eval_samples, n_eval_seeds
            )

            self.search_history.append({
                'iter': i + 1,
                'objective': metrics['objective'],
                **metrics,
            })

            if metrics['objective'] < self.best_objective:
                self.best_objective = metrics['objective']
                self.best_attacker_acc = metrics['attacker_accuracy']
                self.best_params = params
                print(f"    Iter {i+1}: NEW BEST obj={metrics['objective']:.4f}, "
                      f"attacker_acc={metrics['attacker_accuracy']:.3f}, "
                      f"entropy={metrics['phase_entropy']:.3f}, "
                      f"lf_ratio={metrics['lowfreq_energy_ratio']:.3f}")
            else:
                if (i + 1) % 10 == 0:
                    print(f"    Iter {i+1}: obj={metrics['objective']:.4f}, "
                          f"attacker_acc={metrics['attacker_accuracy']:.3f}")

        # 最终评估
        best_metrics = self.evaluate_params(
            self.best_params, attacker_model, device, n_eval_samples * 2, max(n_eval_seeds, 2)
        )

        return {
            'best_params': self.best_params,
            'best_metrics': best_metrics,
            'search_history': self.search_history,
        }
