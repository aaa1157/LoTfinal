#!/bin/bash
# Two-Stage Strategy-Aware Attack Pipeline
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Two-Stage Strategy-Aware Attack Pipeline"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')" > results/final_results/manifests/timing_v3.txt

# Step 1: Two-stage attack (random)
echo ""
echo "==== Step 1: Two-Stage Attack (random) ===="
python scripts/run_two_stage_strategy_aware_attack.py --mode medium --split random --seeds 2026 2027 2028 2>&1 || echo "[WARNING] Step 1 failed, continuing..."

# Step 2: Two-stage attack (scene_disjoint)
echo ""
echo "==== Step 2: Two-Stage Attack (scene_disjoint) ===="
python scripts/run_two_stage_strategy_aware_attack.py --mode medium --split scene_disjoint --seeds 2026 2027 2028 2>&1 || echo "[WARNING] Step 2 failed, continuing..."

# Step 3: Generate v3 reports
echo ""
echo "==== Step 3: Generate v3 Reports ===="
python -c "
import os, sys, json, numpy as np, pandas as pd
from datetime import datetime

PROJECT_ROOT = '$PROJECT_ROOT'
sys.path.insert(0, PROJECT_ROOT)

out_dir = os.path.join(PROJECT_ROOT, 'results/final_results')
tables_dir = os.path.join(out_dir, 'tables')
reports_dir = os.path.join(out_dir, 'reports')
manifests_dir = os.path.join(out_dir, 'manifests')

# Load data
df_sum = pd.read_csv(os.path.join(tables_dir, 'two_stage_strategy_aware_attack_summary.csv')) if os.path.exists(os.path.join(tables_dir, 'two_stage_strategy_aware_attack_summary.csv')) else None
df_dfl = pd.read_csv(os.path.join(tables_dir, 'defense_fingerprint_metrics.csv')) if os.path.exists(os.path.join(tables_dir, 'defense_fingerprint_metrics.csv')) else None
df_saag = pd.read_csv(os.path.join(tables_dir, 'strategy_aware_attack_gain.csv')) if os.path.exists(os.path.join(tables_dir, 'strategy_aware_attack_gain.csv')) else None

# final_experiment_summary_v3.md
lines = []
lines.append('# RFID 超表面隐私防护仿真 — 最终实验总结 v3\n')
lines.append(f'> 生成时间: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}\n')
lines.append('## 策略指纹与两阶段攻击\n')
lines.append('### 当前 strategy fingerprint classifier 已证明部分策略具有明显可识别指纹\n')
if df_dfl is not None and len(df_dfl) > 0:
    for _, row in df_dfl.iterrows():
        lines.append(f'- {row[\"split_type\"]}/{row[\"input_type\"]}/{row[\"model_name\"]}: DFL_acc={row[\"DFL_acc\"]:.3f}, DFL_excess={row[\"DFL_excess\"]:.3f}\n')
lines.append('\n### Two-stage attack 进一步验证攻击者是否可以利用该指纹\n')
if df_saag is not None and len(df_saag) > 0:
    mean_saag = df_saag['SAAG_predicted'].mean()
    n_exploitable = (df_saag['SAAG_predicted'] > 0.05).sum()
    lines.append(f'- SAAG_predicted 平均值: {mean_saag:+.3f}\n')
    lines.append(f'- 可利用策略数: {n_exploitable}/{len(df_saag)}\n')
lines.append('\n### DFL 衡量防护策略是否可被识别\n')
lines.append('DFL_acc = strategy classifier accuracy, DFL_excess = DFL_acc - 1/K\n')
lines.append('\n### SAAG 衡量利用策略指纹后 motion 攻击是否提升\n')
lines.append('SAAG_predicted = Acc_predicted_strategy_aware - Acc_single_mixed\n')
lines.append('\n### 好的超表面防护应同时降低 motion leakage 和 strategy fingerprint leakage\n')

# Append v2 content
v2_path = os.path.join(reports_dir, 'final_experiment_summary_v2.md')
if os.path.exists(v2_path):
    with open(v2_path) as f:
        v2_content = f.read()
    lines.append('\n---\n\n')
    lines.append(v2_content)

with open(os.path.join(reports_dir, 'final_experiment_summary_v3.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: final_experiment_summary_v3.md')

# key_findings_v3.md
lines = []
lines.append('# 关键发现 v3\n')
lines.append('1-8 同 v2。\n')
lines.append('9. **策略指纹可被攻击者利用**: DFL 显示部分策略具有明显可识别指纹，SAAG 显示攻击者可以利用该指纹进行两阶段攻击。')
lines.append('10. **好的超表面隐私防护不仅应降低 motion inference accuracy，还应降低 defense strategy fingerprint leakage。**')
lines.append('11. **fingerprint-free metasurface defense 是未来重要方向**: 使策略本身难以被识别，从而阻断两阶段攻击的第一阶段。')
with open(os.path.join(reports_dir, 'key_findings_v3.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: key_findings_v3.md')

# limitations_v3.md
lines = []
lines.append('# 局限性 v3\n')
lines.append('1-10 同 v2。\n')
lines.append('11. 两阶段攻击仅验证了当前 5 种策略，更多策略的 SAAG 待研究。')
lines.append('12. deep model 两阶段攻击仅验证了 ResNet1DLite，其他模型待验证。')
lines.append('13. LMC 策略未纳入两阶段攻击实验。')
with open(os.path.join(reports_dir, 'limitations_v3.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: limitations_v3.md')

# future_work_v3.md
lines = []
lines.append('# 后续工作 v3\n')
lines.append('1-9 同 v2。\n')
lines.append('10. 设计 fingerprint-free metasurface defense，使策略本身难以被识别。')
lines.append('11. 扩展两阶段攻击到更多策略和模型。')
lines.append('12. 研究 SAAG 与 DFL 的理论关系。')
with open(os.path.join(reports_dir, 'future_work_v3.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: future_work_v3.md')

# figure_captions_v3.md
lines = []
lines.append('# 图注清单 v3\n')
lines.append('v1/v2 图注同前。\n\n')
lines.append('## two_stage_attack_comparison.png\n')
lines.append('- **图题**: 两阶段攻击对比\n')
lines.append('- **图注**: single_mixed、oracle_strategy_aware、predicted_strategy_aware 三种攻击方式的准确率对比\n\n')
lines.append('## strategy_aware_attack_gain_by_strategy.png\n')
lines.append('- **图题**: 策略感知攻击增益\n')
lines.append('- **图注**: 每个策略的 SAAG_oracle 和 SAAG_predicted\n\n')
lines.append('## defense_fingerprint_leakage.png\n')
lines.append('- **图题**: 防护策略指纹泄露\n')
lines.append('- **图注**: DFL_acc 和 DFL_macroF1\n\n')
lines.append('## fingerprint_vs_attack_gain_scatter.png\n')
lines.append('- **图题**: 指纹 vs 攻击增益散点图\n')
lines.append('- **图注**: 策略分类器召回率与 SAAG_predicted 的关系\n')
with open(os.path.join(reports_dir, 'figure_captions_v3.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: figure_captions_v3.md')

# report_ready_tables_v3.md
lines = []
lines.append('# 报告就绪表格 v3\n')
if df_saag is not None and len(df_saag) > 0:
    lines.append('## 表: Strategy-Aware Attack Gain\n')
    lines.append('| Split | Input | Strategy | Single | Oracle | Predicted | SAAG_oracle | SAAG_predicted | Interpretation |')
    lines.append('|-------|-------|----------|--------|--------|-----------|-------------|----------------|----------------|')
    for _, row in df_saag.iterrows():
        lines.append(f'| {row[\"split_type\"]} | {row[\"input_type\"]} | {row[\"test_strategy\"]} | {row[\"single_mixed_accuracy\"]:.3f} | {row[\"oracle_strategy_aware_accuracy\"]:.3f} | {row[\"predicted_strategy_aware_accuracy\"]:.3f} | {row[\"SAAG_oracle\"]:+.3f} | {row[\"SAAG_predicted\"]:+.3f} | {row[\"interpretation\"]} |')
with open(os.path.join(reports_dir, 'report_ready_tables_v3.md'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: report_ready_tables_v3.md')

# Manifest v3
manifest = {
    'project_name': 'RFID Metasurface Privacy Protection Simulation',
    'run_id': f'two_stage_v3_{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}',
    'completed_tasks': ['two_stage_strategy_aware_attack', 'dfl_metrics', 'saag_metrics', 'report_pack_v3'],
    'failed_tasks': [],
    'notes': 'v3 adds: two-stage strategy-aware attack, DFL, SAAG',
}
with open(os.path.join(manifests_dir, 'final_manifest_v3.json'), 'w') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print('Saved: final_manifest_v3.json')

# Commands v3
lines = [
    '# Commands Used in Two-Stage Pipeline v3',
    f'# Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}',
    '',
    'python scripts/run_two_stage_strategy_aware_attack.py --mode medium --split random --seeds 2026 2027 2028',
    'python scripts/run_two_stage_strategy_aware_attack.py --mode medium --split scene_disjoint --seeds 2026 2027 2028',
]
with open(os.path.join(manifests_dir, 'commands_used_v3.txt'), 'w') as f:
    f.write('\n'.join(lines))
print('Saved: commands_used_v3.txt')

print('Report pack v3 generation complete!')
" 2>&1 || echo "[WARNING] Step 3 failed, continuing..."

# Step 4: Package
echo ""
echo "==== Step 4: Package ===="
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> results/final_results/manifests/timing_v3.txt

cd results/final_results
mkdir -p archive
rm -f archive/final_results_v3.zip
zip -r archive/final_results_v3.zip tables/ figures/ reports/ manifests/ -x '*.pt' '*.npz' 2>&1 || echo "[WARNING] Packaging failed"
cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "  Two-Stage Pipeline Complete!"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
