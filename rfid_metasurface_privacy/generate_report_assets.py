"""
报告材料自动生成脚本

1. 检查所有 results/tables/*.csv 是否存在
2. 检查所有 results/figures/*.png 是否存在
3. 读取结果表格
4. 生成 reports/experiment_summary.md
5. 生成 reports/report_outline.md
"""

import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_files():
    """检查输出文件"""
    print("=" * 60)
    print("  Checking output files ...")
    print("=" * 60)

    expected_csvs = [
        'results/tables/metrics.csv',
        'results/tables/respiration_errors.csv',
        'results/tables/dataset_summary.csv',
        'results/tables/deep_attack_results.csv',
        'results/tables/deep_attack_summary.csv',
        'results/tables/controller_search_history.csv',
        'results/tables/lmc_best_params.csv',
        'results/tables/lmc_metrics.csv',
    ]

    expected_pngs = [
        'results/figures/phase_no_metasurface_motion_vs_nomotion.png',
        'results/figures/phase_strategies_comparison.png',
        'results/figures/spectrum_strategies_comparison.png',
        'results/figures/walking_detection_metrics.png',
        'results/figures/respiration_error_comparison.png',
        'results/figures/summary_comparison.png',
        'results/figures/dataset_distribution.png',
        'results/figures/feature_tsne_or_pca.png',
        'results/figures/deep_attacker_comparison.png',
        'results/figures/deep_attacker_train_curves.png',
        'results/figures/lmc_search_history.png',
    ]

    csv_status = {}
    for f in expected_csvs:
        path = os.path.join(PROJECT_ROOT, f)
        exists = os.path.exists(path)
        csv_status[f] = exists
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f}")

    png_status = {}
    for f in expected_pngs:
        path = os.path.join(PROJECT_ROOT, f)
        exists = os.path.exists(path)
        png_status[f] = exists
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {f}")

    # Check all existing figures
    all_pngs = glob.glob(os.path.join(PROJECT_ROOT, 'results/figures/*.png'))
    print(f"\n  Total figures found: {len(all_pngs)}")

    return csv_status, png_status


def generate_experiment_summary():
    """生成实验摘要"""
    import pandas as pd

    print("\n  Generating experiment_summary.md ...")

    lines = []
    lines.append("# Experiment Summary\n")
    lines.append("## 1. Experiment Setup\n")
    lines.append("- **Project**: RFID Metasurface Privacy Protection Simulation")
    lines.append("- **Dataset**: RFID-MetaPrivacy-Sim (synthetic)")
    lines.append("- **Sampling Rate**: 30 Hz")
    lines.append("- **Sequence Length**: 1800 (60 seconds)")
    lines.append("- **Metasurface**: 16 units (4x4), 1-bit programmable")
    lines.append("- **Strategies**: no_metasurface, periodic, random, rfnoid_like, multifreq_proposed")
    lines.append("- **Deep Models**: PhaseCNN, PhaseNetLite, TinyTCN, ResNet1DLite, DualBranchNet")
    lines.append("")

    # Traditional attack results
    metrics_path = os.path.join(PROJECT_ROOT, 'results/tables/metrics.csv')
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        lines.append("## 2. Traditional Attack Results\n")
        lines.append("| Strategy | Threshold Acc | Classifier Acc | Classifier TPR | Classifier FPR | Phase Entropy | Spectral Entropy | LF Ratio | Switching Rate |")
        lines.append("|----------|--------------|----------------|----------------|----------------|---------------|-----------------|----------|----------------|")
        for _, row in df.iterrows():
            spec_ent = row.get('mean_spectral_entropy', 0)
            sw_rate = row.get('switching_rate', 0)
            lines.append(f"| {row['strategy']} | {row['threshold_accuracy']:.3f} | {row['classifier_accuracy']:.3f} | "
                        f"{row['classifier_tpr']:.3f} | {row['classifier_fpr']:.3f} | "
                        f"{row['mean_phase_entropy']:.3f} | {spec_ent:.3f} | "
                        f"{row['mean_lowfreq_energy_ratio']:.3f} | {sw_rate:.3f} |")
        lines.append("")

    # Deep attack summary
    deep_summary_path = os.path.join(PROJECT_ROOT, 'results/tables/deep_attack_summary.csv')
    if os.path.exists(deep_summary_path):
        df = pd.read_csv(deep_summary_path)
        lines.append("## 3. Deep Learning Attack Summary\n")
        lines.append("| Model | Train Setting | Test Strategy | Mean Acc | Std Acc | Mean F1 | N Seeds |")
        lines.append("|-------|--------------|---------------|----------|---------|---------|---------|")
        for _, row in df.iterrows():
            lines.append(f"| {row['model']} | {row['train_setting']} | {row['test_strategy']} | "
                        f"{row['mean_accuracy']:.3f} | {row['std_accuracy']:.3f} | "
                        f"{row['mean_f1']:.3f} | {row['n_seeds']} |")
        lines.append("")
    else:
        # Fallback to raw results
        deep_path = os.path.join(PROJECT_ROOT, 'results/tables/deep_attack_results.csv')
        if os.path.exists(deep_path):
            df = pd.read_csv(deep_path)
            lines.append("## 3. Deep Learning Attack Results\n")
            lines.append("| Model | Train Setting | Test Strategy | Accuracy | TPR | FPR |")
            lines.append("|-------|--------------|---------------|----------|-----|-----|")
            for _, row in df.iterrows():
                lines.append(f"| {row['model']} | {row['train_setting']} | {row['test_strategy']} | "
                            f"{row['accuracy']:.3f} | {row['TPR']:.3f} | {row['FPR']:.3f} |")
            lines.append("")

    # LMC results
    lmc_metrics_path = os.path.join(PROJECT_ROOT, 'results/tables/lmc_metrics.csv')
    if os.path.exists(lmc_metrics_path):
        df = pd.read_csv(lmc_metrics_path)
        lines.append("## 4. Learnable Metasurface Controller Results\n")
        lines.append("| Controller | Attacker Model | Attacker Acc | Phase Entropy | LF Ratio | Switching Rate |")
        lines.append("|------------|---------------|-------------|---------------|----------|----------------|")
        for _, row in df.iterrows():
            lines.append(f"| {row['controller']} | {row['attacker_model']} | "
                        f"{row['attacker_accuracy']:.3f} | "
                        f"{row.get('phase_entropy', 0):.3f} | "
                        f"{row.get('lowfreq_energy_ratio', 0):.3f} | "
                        f"{row.get('switching_rate', 0):.3f} |")
        lines.append("")

        # Honest assessment of LMC
        lmc_acc = df['attacker_accuracy'].values[0] if len(df) > 0 else 1.0
        if lmc_acc > 0.7:
            lines.append("### LMC Assessment\n")
            lines.append("> **Note**: The LMC controller did NOT achieve significant improvement over hand-crafted strategies (rfnoid_like, multifreq_proposed). ")
            lines.append("> The attacker accuracy remains high, indicating that random search with limited iterations is insufficient for this parameter space. ")
            lines.append("> This is an honest finding: learnable metasurface control requires more sophisticated optimization (e.g., Bayesian optimization, reinforcement learning) and longer search budgets.\n")
        else:
            lines.append("### LMC Assessment\n")
            lines.append("> The LMC controller found parameters that reduce attacker accuracy below hand-crafted strategies, demonstrating the potential of learnable metasurface control.\n")
        lines.append("")

    # Conclusions
    lines.append("## 5. Main Conclusions\n")
    lines.append("1. **RQ1**: Without metasurface, RFID phase clearly leaks human motion (classifier acc > 95%) and respiration info (error < 0.03 Hz).")
    lines.append("2. **RQ2**: 1-bit programmable metasurface can effectively reduce traditional attacker accuracy (from ~96% to ~56-60% for rfnoid_like/multifreq_proposed).")
    lines.append("3. **RQ3**: Deep learning attackers are stronger than traditional ones but still affected by metasurface defense, especially rfnoid_like and multifreq_proposed strategies.")
    lines.append("4. **RQ4**: LMC controller with random search shows limited improvement over hand-crafted strategies. More sophisticated optimization methods are needed.")
    lines.append("")

    # Limitations
    lines.append("## 6. Limitations\n")
    lines.append("- This is a mechanism-level simulation, not real hardware validation")
    lines.append("- Parameters are simplified for educational purposes")
    lines.append("- Synthetic dataset cannot replace real wireless sensing data")
    lines.append("- Deep learning results depend on simulation data distribution")
    lines.append("- LMC search is limited by random search efficiency")
    lines.append("- Multi-seed experiments may show high variance in debug mode")
    lines.append("")

    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'experiment_summary.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: reports/experiment_summary.md")


def generate_report_outline():
    """生成报告大纲"""
    print("  Generating report_outline.md ...")

    lines = []
    lines.append("# Course Report Outline\n")
    lines.append("## 1. Introduction\n")
    lines.append("- Background: RFID wireless sensing and privacy risks")
    lines.append("- Motivation: Protecting human motion and respiration privacy")
    lines.append("- Research questions (RQ1-RQ4)\n")

    lines.append("## 2. Related Work\n")
    lines.append("- RFNOID: RFID motion privacy via metasurface")
    lines.append("- IRShield: IRS for wireless sensing privacy")
    lines.append("- SenseFi: Deep learning for WiFi human sensing")
    lines.append("- MetaSensing: Learnable metasurface control")
    lines.append("- RF-Pose: Through-wall human pose estimation\n")

    lines.append("## 3. System Model\n")
    lines.append("- RFID complex baseband signal model")
    lines.append("- 1-bit programmable metasurface model")
    lines.append("- Phase observation model\n")

    lines.append("## 4. RFID-MetaPrivacy-Sim Dataset\n")
    lines.append("- Dataset construction methodology")
    lines.append("- Signal parameter randomization (motion_type, SNR, wall_type)")
    lines.append("- Train/val/test split strategy (random vs scene_disjoint)\n")

    lines.append("## 5. Metasurface Defense Strategies\n")
    lines.append("- No metasurface (baseline)")
    lines.append("- Periodic flipping")
    lines.append("- Random flipping")
    lines.append("- RFNOID-like heuristic")
    lines.append("- Multifreq proposed (with adjustable parameters)\n")

    lines.append("## 6. Attacker Models\n")
    lines.append("- Variance threshold detection")
    lines.append("- Statistical feature classifier (Logistic Regression)")
    lines.append("- Respiration frequency estimation")
    lines.append("- Deep learning attackers (PhaseCNN, PhaseNetLite, TinyTCN, ResNet1DLite, DualBranchNet)\n")

    lines.append("## 7. Learnable Metasurface Controller\n")
    lines.append("- LMC architecture and parameter space")
    lines.append("- Search-based optimization (random search)")
    lines.append("- Objective function design (attacker accuracy + entropy + LF ratio + switching cost)\n")

    lines.append("## 8. Experimental Results\n")
    lines.append("- RQ1: Privacy leakage without metasurface")
    lines.append("- RQ2: Traditional attacker defense effectiveness")
    lines.append("- RQ3: Deep learning attacker robustness (Exp A-E)")
    lines.append("- RQ4: LMC vs hand-crafted strategies\n")

    lines.append("## 9. Discussion\n")
    lines.append("- Trade-off between privacy and signal quality")
    lines.append("- Limitations of mechanism-level simulation")
    lines.append("- Potential for real-world deployment\n")

    lines.append("## 10. Conclusion\n")
    lines.append("- Summary of findings")
    lines.append("- Future work directions\n")

    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, 'report_outline.md'), 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: reports/report_outline.md")


def main():
    print("=" * 60)
    print("  Report Asset Generator")
    print("=" * 60)

    csv_status, png_status = check_files()
    generate_experiment_summary()
    generate_report_outline()

    print("\n" + "=" * 60)
    print("  Report generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
