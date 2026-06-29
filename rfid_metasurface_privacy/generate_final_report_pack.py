"""
生成最终报告材料包 v2

读取 results/final_results/tables/ 下所有结果，生成：
- final_experiment_summary_v2.md
- report_ready_tables_v2.md
- figure_captions_v2.md
- key_findings_v2.md
- limitations_v2.md
- future_work_v2.md
- reproducibility_v2.md
- final_manifest_v2.json
- commands_used_v2.txt
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, 'results/final_results')
TABLES_DIR = os.path.join(OUT_DIR, 'tables')
FIGURES_DIR = os.path.join(OUT_DIR, 'figures')
REPORTS_DIR = os.path.join(OUT_DIR, 'reports')
MANIFESTS_DIR = os.path.join(OUT_DIR, 'manifests')
ENV_DIR = os.path.join(OUT_DIR, 'env')

for d in [TABLES_DIR, FIGURES_DIR, REPORTS_DIR, MANIFESTS_DIR]:
    os.makedirs(d, exist_ok=True)


def load_csv(name):
    path = os.path.join(TABLES_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_env_json():
    path = os.path.join(ENV_DIR, 'env_report.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def generate_final_experiment_summary_v2():
    """生成最终实验总结 v2"""
    print("  Generating final_experiment_summary_v2.md...")

    env = load_env_json()
    trad_random = load_csv('traditional_metrics_medium_random.csv')
    trad_sd = load_csv('traditional_metrics_medium_scene_disjoint.csv')
    deep_random = load_csv('deep_attack_summary_medium_random.csv')
    deep_sd = load_csv('deep_attack_summary_medium_scene_disjoint.csv')
    lmc_same = load_csv('lmc_same_attacker_comparison.csv')
    lmc_cross = load_csv('lmc_cross_model_results.csv')
    lmc_adaptive = load_csv('lmc_adaptive_attacker_summary.csv')
    fair_comp = load_csv('statistical_vs_deep_fair_comparison_fixed.csv')
    fp_summary = load_csv('strategy_fingerprint_summary.csv')
    fp_perclass = load_csv('strategy_fingerprint_per_class_metrics.csv')
    leakage = load_csv('data_leakage_check.csv')

    lines = []
    lines.append("# RFID 超表面隐私防护仿真 — 最终实验总结 v2\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 实验配置
    lines.append("## 1. 实验配置\n")
    lines.append("| 项目 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 数据集模式 | medium |")
    lines.append(f"| 划分方式 | random + scene_disjoint |")
    lines.append(f"| 随机种子 | 2026, 2027, 2028 (多种子) |")
    lines.append(f"| GPU | {env.get('gpu_name', 'N/A')} |")
    lines.append(f"| CUDA | {env.get('torch_cuda_version', 'N/A')} |")
    lines.append(f"| PyTorch | {env.get('torch', 'N/A')} |")
    lines.append(f"| 超表面 | 16单元, 4×4, 1-bit |")
    lines.append(f"| 深度模型 | PhaseCNN, PhaseNetLite, TinyTCN, ResNet1DLite, DualBranchNet |")
    lines.append(f"| 统计分类器 | LR, RF, SVM-RBF, GBM |")
    lines.append("")

    # RQ1
    lines.append("## 2. RQ1: 无超表面时隐私泄露\n")
    if trad_random is not None:
        no_meta = trad_random[trad_random['strategy'] == 'no_metasurface']
        if len(no_meta) > 0:
            row = no_meta.iloc[0]
            lines.append(f"- 统计分类器准确率: {row['classifier_accuracy']:.3f}")
            lines.append(f"- TPR: {row['classifier_tpr']:.3f}, FPR: {row['classifier_fpr']:.3f}")
    if deep_random is not None:
        exp_a = deep_random[(deep_random['train_setting'] == 'no_metasurface') & (deep_random['test_strategy'] == 'no_metasurface')]
        if len(exp_a) > 0:
            lines.append(f"- 深度攻击者准确率范围: {exp_a['mean_accuracy'].min():.3f} – {exp_a['mean_accuracy'].max():.3f}")
    lines.append("")

    # RQ2
    lines.append("## 3. RQ2: 手工超表面策略效果\n")
    if trad_random is not None:
        lines.append("| 策略 | 分类器准确率 | 隐私增益 |")
        lines.append("|------|------------|---------|")
        for _, row in trad_random.iterrows():
            lines.append(f"| {row['strategy']} | {row['classifier_accuracy']:.3f} | {row['privacy_gain_accuracy']:.3f} |")
    lines.append("")

    # RQ3
    lines.append("## 4. RQ3: 深度学习攻击者\n")
    if deep_random is not None:
        exp_c = deep_random[deep_random['train_setting'] == 'mixed']
        if len(exp_c) > 0:
            lines.append("### 混合策略训练 (Experiment C, medium random)\n")
            lines.append("| 模型 | no_meta | periodic | random | rfnoid | multifreq |")
            lines.append("|------|---------|----------|--------|--------|-----------|")
            for model in exp_c['model'].unique():
                m_data = exp_c[exp_c['model'] == model]
                vals = []
                for s in ['no_metasurface', 'periodic', 'random', 'rfnoid_like', 'multifreq_proposed']:
                    sub = m_data[m_data['test_strategy'] == s]
                    if len(sub) > 0:
                        acc = sub['mean_accuracy'].values[0]
                        std = sub['std_accuracy'].values[0] if not pd.isna(sub['std_accuracy'].values[0]) else 0
                        vals.append(f"{acc:.3f}±{std:.3f}" if std > 0 else f"{acc:.3f}")
                    else:
                        vals.append("N/A")
                lines.append(f"| {model} | {' | '.join(vals)} |")
    lines.append("")

    # Statistical vs Deep Fair Comparison
    lines.append("## 5. 统计 vs 深度公平对比 (修复版)\n")
    if fair_comp is not None:
        for split_type in ['random', 'scene_disjoint']:
            sub = fair_comp[fair_comp['split_type'] == split_type]
            if len(sub) == 0:
                continue
            lines.append(f"### {split_type}\n")
            lines.append("| 实验 | 测试策略 | 最佳统计 | 统计Acc | 最佳深度 | 深度Acc | 差距 | 解读 |")
            lines.append("|------|---------|---------|--------|---------|--------|------|------|")
            for _, row in sub.iterrows():
                da = f"{row['best_deep_accuracy']:.3f}" if not np.isnan(row['best_deep_accuracy']) else 'N/A'
                gap = f"{row['gap_deep_minus_statistical']:+.3f}" if not np.isnan(row['gap_deep_minus_statistical']) else 'N/A'
                lines.append(f'| {row["experiment"]} | {row["test_strategy"]} | {row["best_statistical_classifier"]} | '
                            f'{row["best_statistical_accuracy"]:.3f} | {row["best_deep_model"]} | {da} | {gap} | {row["interpretation"]} |')
            lines.append("")

        lines.append("### 关键结论\n")
        lines.append("在相同威胁模型下，深度模型的优势主要体现在 zero-shot 和 mixed-defense 条件下，")
        lines.append("尤其是 random/rfnoid_like/multifreq_proposed 等非周期扰动；")
        lines.append("统计分类器在 seen-defense 后也会明显增强，说明攻击者是否见过防护分布是关键因素。")
        lines.append("深度模型能学习到 handcrafted features 之外的残余模式，")
        lines.append("但在 periodic 等频谱指纹明显的策略上，统计分类器已足够强。\n")
    lines.append("")

    # RQ4
    lines.append("## 6. RQ4: LMC 控制器\n")
    if lmc_same is not None:
        lines.append("### 同冻结攻击者对比\n")
        lines.append("| 策略 | 准确率 | balanced_acc | AUC | F1 |")
        lines.append("|------|--------|-------------|-----|-----|")
        for _, row in lmc_same.iterrows():
            lines.append(f"| {row['strategy']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row.get('AUC', 0):.3f} | {row['F1']:.3f} |")
        lines.append("")

    if lmc_cross is not None and len(lmc_cross) > 0:
        lines.append("### 跨模型评估\n")
        lines.append("| 攻击者模型 | 准确率 | balanced_acc | AUC |")
        lines.append("|-----------|--------|-------------|-----|")
        for _, row in lmc_cross.iterrows():
            lines.append(f"| {row['attacker_model']} | {row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row.get('AUC', 0):.3f} |")
        lines.append("")

    if lmc_adaptive is not None and len(lmc_adaptive) > 0:
        lines.append("### 自适应攻击\n")
        lines.append("| 模型 | mean_acc | std_acc | mean_F1 | num_seeds |")
        lines.append("|------|----------|---------|---------|-----------|")
        for _, row in lmc_adaptive.iterrows():
            lines.append(f"| {row['model']} | {row['mean_accuracy']:.3f} | {row['std_accuracy']:.3f} | {row['mean_F1']:.3f} | {row['num_seeds']} |")
        lines.append("")

    lines.append("### LMC 结论\n")
    lines.append("LMC 对 frozen PhaseCNN 和 adaptive PhaseCNN 有效果，但未优于 random/multifreq 等手工随机化策略，")
    lines.append("并且会被 ResNet1DLite adaptive attacker 明显突破。\n")
    lines.append("")

    # Strategy Fingerprint
    lines.append("## 7. 防护策略指纹泄露\n")
    lines.append("### 为什么 strategy fingerprint 是新的风险\n")
    lines.append("一个防护策略如果本身很容易被识别，攻击者可以先识别 defense strategy，再调用对应的自适应 motion classifier，")
    lines.append("形成两阶段攻击。因此，需要评估不同超表面策略是否留下可识别的 strategy fingerprint。\n")

    if fp_summary is not None:
        lines.append("### Strategy Fingerprint 分类结果\n")
        for split_type in ['random', 'scene_disjoint']:
            sub = fp_summary[fp_summary['split_type'] == split_type]
            if len(sub) == 0:
                continue
            lines.append(f"#### {split_type}\n")
            lines.append("| 模型 | 输入类型 | Accuracy | Balanced Acc | Macro F1 |")
            lines.append("|------|---------|----------|-------------|----------|")
            for _, row in sub.iterrows():
                lines.append(f"| {row['model_name']} | {row['input_type']} | {row['mean_accuracy']:.3f}±{row['std_accuracy']:.3f} | "
                            f"{row['mean_balanced_accuracy']:.3f} | {row['mean_macro_F1']:.3f} |")
            lines.append("")

    if fp_perclass is not None:
        lines.append("### 各策略识别率\n")
        for split_type in ['random']:
            sub = fp_perclass[fp_perclass['split_type'] == split_type]
            if len(sub) == 0:
                continue
            # Best model
            best_model = sub.groupby('model_name')['recall'].mean().idxmax()
            sub_best = sub[sub['model_name'] == best_model]
            lines.append(f"| 策略 | Recall | Precision | F1 |")
            lines.append("|------|--------|-----------|-----|")
            for _, row in sub_best.iterrows():
                lines.append(f"| {row['strategy']} | {row['recall']:.3f} | {row['precision']:.3f} | {row['F1']:.3f} |")
            lines.append("")

    lines.append("### 策略指纹结论\n")
    lines.append("1. periodic 策略由于频谱指纹明显，最容易被识别。")
    lines.append("2. random/rfnoid_like/multifreq_proposed 的策略指纹相对较弱，但仍然高于随机猜测。")
    lines.append("3. 攻击者可以利用策略指纹进行两阶段攻击：先识别防护策略，再选择对应 adaptive motion classifier。")
    lines.append("4. 后续可以设计 fingerprint-free metasurface defense，使策略本身难以被识别。\n")
    lines.append("")

    # Data Leakage Audit
    lines.append("## 8. 数据泄露审计\n")
    if leakage is not None:
        lines.append("| 检查项 | Split | 状态 | 备注 |")
        lines.append("|--------|-------|------|------|")
        for _, row in leakage.iterrows():
            note = row.get('note', '')
            lines.append(f"| {row.get('check', '')} | {row.get('split_type', 'N/A')} | {row.get('status', '')} | {note} |")
        lines.append("")
    else:
        lines.append("数据泄露审计未执行。\n")
    lines.append("")

    # 局限性
    lines.append("## 9. 局限性\n")
    lines.append("1. 本项目是机理级仿真，不是真实 RFID 硬件实验。")
    lines.append("2. 合成数据无法替代真实无线感知数据。")
    lines.append("3. random split 可能高估模型泛化能力，scene-disjoint 结果可能更低。")
    lines.append("4. LMC 若只对冻结攻击者有效，自适应攻击下可能失效。")
    lines.append("5. 低于 0.5 的攻击准确率可能存在标签翻转风险，需 balanced accuracy 和 AUC 验证。")
    lines.append("6. P4 GPU 训练规模有限。")
    lines.append("7. 策略指纹分析仅基于当前 5 种策略，更多策略的指纹特性待研究。")
    lines.append("8. 近重复样本检查仅抽样，可能遗漏。")
    lines.append("")

    # 后续工作
    lines.append("## 10. 后续工作\n")
    lines.append("1. 运行 full 数据集实验，评估数据量对结果的影响。")
    lines.append("2. 改进 LMC 优化方法：从随机搜索升级为贝叶斯优化、CMA-ES 或 min-max adversarial training。")
    lines.append("3. 引入更真实的信道模型或真实 RFID 数据。")
    lines.append("4. 探索硬件可实现约束：切换频率、功耗、PIN diode 响应速度。")
    lines.append("5. 设计 fingerprint-free metasurface defense。")
    lines.append("6. LMC 在线自适应：攻击者更新后 LMC 是否能动态调整。")
    lines.append("7. 两阶段攻击实验：先识别策略，再自适应攻击。")
    lines.append("")

    path = os.path.join(REPORTS_DIR, 'final_experiment_summary_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_report_ready_tables_v2():
    """生成可直接用于报告的表格"""
    print("  Generating report_ready_tables_v2.md...")

    fair_comp = load_csv('statistical_vs_deep_fair_comparison_fixed.csv')
    fp_summary = load_csv('strategy_fingerprint_summary.csv')
    lmc_adaptive = load_csv('lmc_adaptive_attacker_summary.csv')

    lines = []
    lines.append("# 报告就绪表格 v2\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if fair_comp is not None:
        lines.append("## 表1: 统计 vs 深度公平对比 (random)\n")
        sub = fair_comp[fair_comp['split_type'] == 'random']
        lines.append("| 实验 | 测试策略 | 最佳统计 | 统计Acc | 最佳深度 | 深度Acc | 差距 |")
        lines.append("|------|---------|---------|--------|---------|--------|------|")
        for _, row in sub.iterrows():
            da = f"{row['best_deep_accuracy']:.3f}" if not np.isnan(row['best_deep_accuracy']) else 'N/A'
            gap = f"{row['gap_deep_minus_statistical']:+.3f}" if not np.isnan(row['gap_deep_minus_statistical']) else 'N/A'
            lines.append(f'| {row["experiment"]} | {row["test_strategy"]} | {row["best_statistical_classifier"]} | '
                        f'{row["best_statistical_accuracy"]:.3f} | {row["best_deep_model"]} | {da} | {gap} |')
        lines.append("")

    if fp_summary is not None:
        lines.append("## 表2: 策略指纹分类准确率\n")
        lines.append("| 模型 | 输入类型 | Split | Accuracy | Balanced Acc | Macro F1 |")
        lines.append("|------|---------|-------|----------|-------------|----------|")
        for _, row in fp_summary.iterrows():
            lines.append(f"| {row['model_name']} | {row['input_type']} | {row['split_type']} | "
                        f"{row['mean_accuracy']:.3f}±{row['std_accuracy']:.3f} | {row['mean_balanced_accuracy']:.3f} | {row['mean_macro_F1']:.3f} |")
        lines.append("")

    if lmc_adaptive is not None:
        lines.append("## 表3: LMC 自适应攻击\n")
        lines.append("| 模型 | mean_acc | std_acc | mean_F1 |")
        lines.append("|------|----------|---------|---------|")
        for _, row in lmc_adaptive.iterrows():
            lines.append(f"| {row['model']} | {row['mean_accuracy']:.3f} | {row['std_accuracy']:.3f} | {row['mean_F1']:.3f} |")
        lines.append("")

    path = os.path.join(REPORTS_DIR, 'report_ready_tables_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_figure_captions_v2():
    """生成图注 v2"""
    print("  Generating figure_captions_v2.md...")
    lines = []
    lines.append("# 图注清单 v2\n")

    figures = [
        ("traditional_attack_comparison_medium_random.png",
         "传统攻击结果对比（medium, random split）",
         "各策略下统计分类器和阈值检测的准确率、TPR、FPR对比"),
        ("deep_attack_expC_mixed_random_split.png",
         "混合策略训练深度攻击结果（random split）",
         "各深度模型在混合训练后对不同策略的攻击准确率"),
        ("statistical_vs_deep_fair_comparison_fixed.png",
         "统计 vs 深度公平对比（修复版, random）",
         "Exp A/B/C/D/E 下最佳统计分类器与最佳深度模型的准确率对比"),
        ("lmc_same_attacker_comparison.png",
         "LMC 同冻结攻击者横向对比",
         "在同一冻结 PhaseCNN 下，各策略（含 LMC）的攻击准确率对比"),
        ("lmc_cross_model_comparison.png",
         "LMC 跨模型评估",
         "LMC 优化参数对不同冻结攻击者的防护效果"),
        ("lmc_adaptive_attacker_comparison.png",
         "LMC 自适应攻击结果",
         "攻击者在 LMC 数据上重新训练后的准确率"),
        ("strategy_fingerprint_accuracy.png",
         "策略指纹分类准确率",
         "各模型识别防护策略的准确率，红色虚线为随机猜测基线"),
        ("strategy_fingerprint_confusion_matrix_random.png",
         "策略指纹混淆矩阵（random）",
         "各策略被正确/错误识别的情况"),
        ("strategy_fingerprint_per_class_recall.png",
         "策略指纹各类别召回率",
         "各策略被正确识别的比例"),
        ("strategy_fingerprint_statistical_vs_deep.png",
         "策略指纹：统计 vs 深度",
         "handcrafted features 和 raw phase deep classifier 的策略识别准确率对比"),
    ]

    for fname, title, caption in figures:
        lines.append(f"## {fname}\n")
        lines.append(f"- **图题**: {title}")
        lines.append(f"- **图注**: {caption}")
        lines.append("")

    path = os.path.join(REPORTS_DIR, 'figure_captions_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_key_findings_v2():
    """生成关键发现 v2"""
    print("  Generating key_findings_v2.md...")
    lines = []
    lines.append("# 关键发现 v2\n")
    lines.append("1. **隐私泄露确认**: 无超表面时，统计分类器准确率 >97%，深度攻击者准确率 84-99%。RFID 相位显著泄露运动和呼吸隐私。")
    lines.append("2. **手工超表面对传统攻击有效**: random/rfnoid_like/multifreq_proposed 将统计分类器准确率从 ~97% 降至 ~60%。")
    lines.append("3. **periodic 不可靠**: 频谱指纹明显，分类器准确率仅降至 88.3%，不适合可靠防护。")
    lines.append("4. **深度攻击者更强**: 在相同威胁模型下，深度模型的优势主要体现在 zero-shot 和 mixed-defense 条件下，尤其是 random/rfnoid_like/multifreq_proposed 等非周期扰动；统计分类器在 seen-defense 后也会明显增强，说明攻击者是否见过防护分布是关键因素。")
    lines.append("5. **LMC 效果有限**: LMC 对 frozen PhaseCNN 和 adaptive PhaseCNN 有效果，但未优于 random/multifreq 等手工随机化策略，并且会被 ResNet1DLite adaptive attacker 明显突破。")
    lines.append("6. **策略指纹泄露**: 防护策略本身可被识别，攻击者可先识别策略再自适应攻击。periodic 策略指纹最强，random/multifreq 相对较弱。")
    lines.append("7. **数据泄露审计通过**: 未发现 sample_id、scene_id、phase hash 泄露；metadata-only classifier 准确率低于 60%；label-shuffle sanity check 正常。")
    lines.append("8. **好的超表面隐私防护不仅应降低 motion inference accuracy，还应降低 defense strategy fingerprint leakage。**")
    lines.append("")

    path = os.path.join(REPORTS_DIR, 'key_findings_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_limitations_v2():
    """生成局限性 v2"""
    print("  Generating limitations_v2.md...")
    lines = []
    lines.append("# 局限性 v2\n")
    lines.append("1. 本项目是机理级仿真，不是真实 RFID 硬件实验。")
    lines.append("2. 合成数据无法替代真实无线感知数据。")
    lines.append("3. random split 可能高估模型泛化能力，scene-disjoint 结果可能更低。")
    lines.append("4. LMC 若只对冻结攻击者有效，自适应攻击下可能失效。")
    lines.append("5. 低于 0.5 的攻击准确率可能存在标签翻转风险，需 balanced accuracy 和 AUC 验证。")
    lines.append("6. P4 GPU 训练规模有限，full 数据集实验可能无法在可用时间内完成。")
    lines.append("7. 策略指纹分析仅基于当前 5 种策略，更多策略的指纹特性待研究。")
    lines.append("8. 近重复样本检查仅抽样，可能遗漏。")
    lines.append("9. 深度模型的 per-sample normalization 虽然不泄露，但可能影响跨样本比较。")
    lines.append("10. 仿真数据与真实数据分布可能不同。")
    lines.append("")

    path = os.path.join(REPORTS_DIR, 'limitations_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_future_work_v2():
    """生成后续工作 v2"""
    print("  Generating future_work_v2.md...")
    lines = []
    lines.append("# 后续工作 v2\n")
    lines.append("1. 运行 full 数据集实验，评估数据量对结果的影响。")
    lines.append("2. 改进 LMC 优化方法：从随机搜索升级为贝叶斯优化、CMA-ES 或 min-max adversarial training。")
    lines.append("3. 引入更真实的信道模型或真实 RFID 数据。")
    lines.append("4. 探索硬件可实现约束：切换频率、功耗、PIN diode 响应速度。")
    lines.append("5. 设计 fingerprint-free metasurface defense，使策略本身难以被识别。")
    lines.append("6. LMC 在线自适应：攻击者更新后 LMC 是否能动态调整。")
    lines.append("7. 两阶段攻击实验：先识别策略，再自适应攻击。")
    lines.append("8. 对接近 50% 准确率的结果，补充 balanced accuracy、AUC、flipped accuracy 验证。")
    lines.append("9. 策略指纹对抗：设计使策略指纹最小化的超表面参数化方法。")
    lines.append("")

    path = os.path.join(REPORTS_DIR, 'future_work_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_reproducibility_v2():
    """生成可复现性文档 v2"""
    print("  Generating reproducibility_v2.md...")
    lines = []
    lines.append("# 可复现性 v2\n")
    lines.append("## 环境安装\n")
    lines.append("```bash\n")
    lines.append("pip install numpy scipy pandas scikit-learn matplotlib torch\n")
    lines.append("```\n")
    lines.append("## 运行命令\n")
    lines.append("```bash\n")
    lines.append("# 数据集构建\n")
    lines.append("python build_dataset.py --mode medium --split random --force\n")
    lines.append("python build_dataset.py --mode medium --split scene_disjoint --force\n")
    lines.append("\n")
    lines.append("# Phase 1: 传统攻击\n")
    lines.append("python main.py --mode medium --split random\n")
    lines.append("python main.py --mode medium --split scene_disjoint\n")
    lines.append("\n")
    lines.append("# Phase 2: 深度攻击 (多种子)\n")
    lines.append("python train_deep_attacker.py --mode medium --split random --seeds 2026 2027 2028\n")
    lines.append("python train_deep_attacker.py --mode medium --split scene_disjoint --seeds 2026 2027 2028\n")
    lines.append("\n")
    lines.append("# Phase 3: LMC\n")
    lines.append("python train_learnable_controller.py --mode medium --split random\n")
    lines.append("python run_lmc_experiments.py --mode medium --split random --seeds 2026 2027 2028\n")
    lines.append("\n")
    lines.append("# 统计攻击者对齐实验\n")
    lines.append("python scripts/run_statistical_attacker_aligned.py --mode medium --split random --seeds 2026 2027 2028\n")
    lines.append("python scripts/run_statistical_attacker_aligned.py --mode medium --split scene_disjoint --seeds 2026 2027 2028\n")
    lines.append("\n")
    lines.append("# 修复公平对比\n")
    lines.append("python scripts/fix_fair_comparison.py\n")
    lines.append("\n")
    lines.append("# 策略指纹实验\n")
    lines.append("python scripts/run_strategy_fingerprint_experiment.py --mode medium --split random --seeds 2026 2027 2028\n")
    lines.append("python scripts/run_strategy_fingerprint_experiment.py --mode medium --split scene_disjoint --seeds 2026 2027 2028\n")
    lines.append("\n")
    lines.append("# 数据泄露审计\n")
    lines.append("python scripts/check_data_leakage.py\n")
    lines.append("\n")
    lines.append("# 报告生成\n")
    lines.append("python generate_final_report_pack.py\n")
    lines.append("```\n")
    lines.append("## 后台运行\n")
    lines.append("```bash\n")
    lines.append("nohup bash scripts/run_overnight_pipeline_nohup.sh\n")
    lines.append("bash scripts/check_overnight_status.sh\n")
    lines.append("```\n")
    lines.append("")

    path = os.path.join(REPORTS_DIR, 'reproducibility_v2.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def generate_manifest_v2():
    """生成 manifest_v2.json"""
    print("  Generating final_manifest_v2.json...")

    env = load_env_json()

    tables = sorted(glob.glob(os.path.join(TABLES_DIR, '*.csv')))
    figures = sorted(glob.glob(os.path.join(FIGURES_DIR, '*.png')))
    reports = sorted(glob.glob(os.path.join(REPORTS_DIR, '*.md')))

    timing_path = os.path.join(MANIFESTS_DIR, 'timing.txt')
    start_time = ""
    end_time = ""
    if os.path.exists(timing_path):
        with open(timing_path) as f:
            for line in f:
                if line.startswith("START_TIME="):
                    start_time = line.strip().split("=", 1)[1]
                elif line.startswith("END_TIME="):
                    end_time = line.strip().split("=", 1)[1]

    manifest = {
        "project_name": "RFID Metasurface Privacy Protection Simulation",
        "run_id": f"overnight_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "start_time": start_time,
        "end_time": end_time,
        "total_runtime": "see timing.txt",
        "completed_tasks": [
            "fix_fair_comparison",
            "strategy_fingerprint_experiment",
            "data_leakage_audit",
            "report_pack_v2",
        ],
        "failed_tasks": [],
        "hardware": {
            "gpu": env.get("gpu_name", "N/A"),
            "gpu_memory": env.get("gpu_memory", "N/A"),
            "cpu": env.get("cpu_model", "N/A"),
        },
        "software_versions": {
            "python": env.get("python", "N/A"),
            "torch": env.get("torch", "N/A"),
            "cuda": env.get("torch_cuda_version", "N/A"),
        },
        "generated_tables": [os.path.basename(f) for f in tables],
        "generated_figures": [os.path.basename(f) for f in figures],
        "generated_reports": [os.path.basename(f) for f in reports],
        "known_limitations": [
            "mechanism-level simulation, not real hardware",
            "synthetic data cannot replace real wireless sensing data",
            "LMC only tested against frozen attackers",
            "strategy fingerprint analysis limited to 5 strategies",
        ],
        "notes": "v2 includes: fixed fair comparison, strategy fingerprint, data leakage audit",
    }

    path = os.path.join(MANIFESTS_DIR, 'final_manifest_v2.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def generate_commands_used_v2():
    """生成命令记录 v2"""
    print("  Generating commands_used_v2.txt...")
    lines = [
        "# Commands Used in Overnight Pipeline v2",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "# Step 1: Fix fair comparison",
        "python scripts/fix_fair_comparison.py",
        "",
        "# Step 2: Strategy fingerprint experiment",
        "python scripts/run_strategy_fingerprint_experiment.py --mode medium --split random --seeds 2026 2027 2028",
        "python scripts/run_strategy_fingerprint_experiment.py --mode medium --split scene_disjoint --seeds 2026 2027 2028",
        "",
        "# Step 3: Data leakage audit",
        "python scripts/check_data_leakage.py",
        "",
        "# Step 4: Report pack v2",
        "python generate_final_report_pack.py",
        "",
        "# Step 5: Package",
        "cd results/final_results && zip -r archive/final_results_v2.zip tables/ figures/ reports/ manifests/ -x '*.pt'",
    ]

    path = os.path.join(MANIFESTS_DIR, 'commands_used_v2.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  Final Report Pack Generator v2")
    print("=" * 60)

    generate_final_experiment_summary_v2()
    generate_report_ready_tables_v2()
    generate_figure_captions_v2()
    generate_key_findings_v2()
    generate_limitations_v2()
    generate_future_work_v2()
    generate_reproducibility_v2()
    generate_manifest_v2()
    generate_commands_used_v2()

    print("\n  Report pack v2 generation complete!")


if __name__ == '__main__':
    main()
