#!/bin/bash
# Overnight Pipeline v2
# 1. Fix fair comparison
# 2. Strategy Fingerprint Classifier
# 3. Data Leakage Audit
# 4. Report Pack v2
# 5. Manifest & Package

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="results/final_results/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Overnight Pipeline v2"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Record start time
echo "START_TIME=$(date '+%Y-%m-%d %H:%M:%S')" > results/final_results/manifests/timing.txt

# ---- Step 1: Fix fair comparison ----
echo ""
echo "==== Step 1: Fix Statistical vs Deep Fair Comparison ===="
python scripts/fix_fair_comparison.py 2>&1 || echo "[WARNING] Step 1 failed, continuing..."

# ---- Step 2: Strategy Fingerprint (random) ----
echo ""
echo "==== Step 2a: Strategy Fingerprint (random) ===="
python scripts/run_strategy_fingerprint_experiment.py --mode medium --split random --seeds 2026 2027 2028 2>&1 || echo "[WARNING] Step 2a failed, continuing..."

# ---- Step 2b: Strategy Fingerprint (scene_disjoint) ----
echo ""
echo "==== Step 2b: Strategy Fingerprint (scene_disjoint) ===="
python scripts/run_strategy_fingerprint_experiment.py --mode medium --split scene_disjoint --seeds 2026 2027 2028 2>&1 || echo "[WARNING] Step 2b failed, continuing..."

# ---- Step 3: Data Leakage Audit ----
echo ""
echo "==== Step 3: Data Leakage Audit ===="
python scripts/check_data_leakage.py 2>&1 || echo "[WARNING] Step 3 failed, continuing..."

# ---- Step 4: Report Pack v2 ----
echo ""
echo "==== Step 4: Generate Report Pack v2 ===="
python generate_final_report_pack.py 2>&1 || echo "[WARNING] Step 4 failed, continuing..."

# ---- Step 5: Manifest & Package ----
echo ""
echo "==== Step 5: Package ===="
# Record end time
echo "END_TIME=$(date '+%Y-%m-%d %H:%M:%S')" >> results/final_results/manifests/timing.txt

# Package
cd results/final_results
mkdir -p archive
rm -f archive/final_results_v2.zip
zip -r archive/final_results_v2.zip tables/ figures/ reports/ manifests/ -x '*.pt' '*.npz' 2>&1 || echo "[WARNING] Packaging failed"
cd "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "  Overnight Pipeline v2 Complete!"
echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
