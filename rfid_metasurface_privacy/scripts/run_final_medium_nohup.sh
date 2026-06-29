#!/bin/bash
# ============================================================
# Final Experiment Pipeline - Medium Mode
# Supports nohup background running
# ============================================================

set -e

cd /root/finalexam/rfid_metasurface_privacy

OUT_BASE="results/final_results"
LOG_DIR="$OUT_BASE/logs"
mkdir -p "$LOG_DIR" "$OUT_BASE/tables" "$OUT_BASE/figures" "$OUT_BASE/reports" "$OUT_BASE/env" "$OUT_BASE/manifests" "$OUT_BASE/checkpoints" "$OUT_BASE/archive"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/final_medium_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/final_medium.pid"

# Record start time
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')" > "$OUT_BASE/manifests/timing.txt"

log() { echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

log "============================================================"
log "  Final Experiment Pipeline - Medium Mode"
log "  Log: $LOG_FILE"
log "============================================================"

# ============================================================
# Step 0: Collect environment info
# ============================================================
log "[Step 0] Collecting environment info..."
bash scripts/collect_env_info.sh 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 1: Build medium random dataset
# ============================================================
log "[Step 1] Building medium random dataset..."
python -u build_dataset.py --mode medium --split random --force 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2: Run Phase 1 - Traditional attacks (medium random)
# ============================================================
log "[Step 2] Running Phase 1 - Traditional attacks (medium random)..."
python -u main.py --mode medium --split random 2>&1 | tee -a "$LOG_FILE"

# Copy results to final_results
cp results/tables/metrics.csv "$OUT_BASE/tables/traditional_metrics_medium_random.csv" 2>/dev/null || true
cp results/tables/respiration_errors.csv "$OUT_BASE/tables/respiration_errors_medium_random.csv" 2>/dev/null || true

# ============================================================
# Step 3: Run Phase 2 - Deep attacks (medium random, seeds 2026/2027/2028)
# ============================================================
log "[Step 3] Running Phase 2 - Deep attacks (medium random, seeds 2026/2027/2028)..."
python -u train_deep_attacker.py --mode medium --split random --seeds 2026 2027 2028 2>&1 | tee -a "$LOG_FILE"

cp results/tables/deep_attack_results.csv "$OUT_BASE/tables/deep_attack_results_medium_random.csv" 2>/dev/null || true
cp results/tables/deep_attack_summary.csv "$OUT_BASE/tables/deep_attack_summary_medium_random.csv" 2>/dev/null || true

# ============================================================
# Step 4: Build medium scene_disjoint dataset
# ============================================================
log "[Step 4] Building medium scene_disjoint dataset..."
python -u build_dataset.py --mode medium --split scene_disjoint --force 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 5: Run Phase 1 - Traditional attacks (medium scene_disjoint)
# ============================================================
log "[Step 5] Running Phase 1 - Traditional attacks (medium scene_disjoint)..."
python -u main.py --mode medium --split scene_disjoint 2>&1 | tee -a "$LOG_FILE"

cp results/tables/metrics.csv "$OUT_BASE/tables/traditional_metrics_medium_scene_disjoint.csv" 2>/dev/null || true
cp results/tables/respiration_errors.csv "$OUT_BASE/tables/respiration_errors_medium_scene_disjoint.csv" 2>/dev/null || true

# ============================================================
# Step 6: Run Phase 2 - Deep attacks (medium scene_disjoint, seeds 2026/2027/2028)
# ============================================================
log "[Step 6] Running Phase 2 - Deep attacks (medium scene_disjoint, seeds 2026/2027/2028)..."
python -u train_deep_attacker.py --mode medium --split scene_disjoint --seeds 2026 2027 2028 2>&1 | tee -a "$LOG_FILE"

cp results/tables/deep_attack_results.csv "$OUT_BASE/tables/deep_attack_results_medium_scene_disjoint.csv" 2>/dev/null || true
cp results/tables/deep_attack_summary.csv "$OUT_BASE/tables/deep_attack_summary_medium_scene_disjoint.csv" 2>/dev/null || true

# ============================================================
# Step 7: Run Phase 3 - LMC (medium random)
# ============================================================
log "[Step 7] Running Phase 3 - LMC search (medium random)..."
python -u train_learnable_controller.py --mode medium --split random 2>&1 | tee -a "$LOG_FILE"

cp results/tables/controller_search_history.csv "$OUT_BASE/tables/lmc_search_history.csv" 2>/dev/null || true
cp results/tables/lmc_best_params.csv "$OUT_BASE/tables/lmc_best_params.csv" 2>/dev/null || true
cp results/tables/lmc_metrics.csv "$OUT_BASE/tables/lmc_metrics.csv" 2>/dev/null || true

# ============================================================
# Step 8: LMC same-attacker comparison
# ============================================================
log "[Step 8] Running LMC same-attacker comparison..."
python -u run_lmc_experiments.py --mode medium --split random --experiment same_attacker 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 9: LMC cross-model evaluation
# ============================================================
log "[Step 9] Running LMC cross-model evaluation..."
python -u run_lmc_experiments.py --mode medium --split random --experiment cross_model 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 10: LMC adaptive attacker
# ============================================================
log "[Step 10] Running LMC adaptive attacker..."
python -u run_lmc_experiments.py --mode medium --split random --seeds 2026 2027 2028 --experiment adaptive 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 11: Generate final report pack
# ============================================================
log "[Step 11] Generating final report pack..."
python -u generate_final_report_pack.py 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 12: Package final_results.zip
# ============================================================
log "[Step 12] Packaging final_results.zip..."
cd results/final_results
zip -r archive/final_results.zip env/ logs/ tables/ figures/ reports/ manifests/ -x "checkpoints/*" "*.pt" 2>&1 | tee -a "$LOG_FILE"
cd /root/finalexam/rfid_metasurface_privacy

# Record end time
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> "$OUT_BASE/manifests/timing.txt"

log "============================================================"
log "  Final Experiment Pipeline COMPLETE!"
log "============================================================"
