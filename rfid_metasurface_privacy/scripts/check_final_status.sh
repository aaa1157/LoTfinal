#!/bin/bash
# Check final experiment status

PID_FILE="results/final_results/logs/final_medium.pid"
LOG_DIR="results/final_results/logs"

echo "============================================================"
echo "  Final Experiment Status Check"
echo "============================================================"

# Check PID
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "  Process $PID: RUNNING"
    else
        echo "  Process $PID: NOT RUNNING"
    fi
else
    echo "  No PID file found"
fi

# GPU status
echo ""
echo "  GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  nvidia-smi not available"

# Latest log
echo ""
echo "  Latest log (last 80 lines):"
LATEST_LOG=$(ls -t "$LOG_DIR"/final_medium_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    tail -80 "$LATEST_LOG"
else
    echo "  No log files found"
fi

# Key files
echo ""
echo "  Generated key files:"
for f in \
    results/final_results/tables/traditional_metrics_medium_random.csv \
    results/final_results/tables/traditional_metrics_medium_scene_disjoint.csv \
    results/final_results/tables/deep_attack_results_medium_random.csv \
    results/final_results/tables/deep_attack_results_medium_scene_disjoint.csv \
    results/final_results/tables/lmc_same_attacker_comparison.csv \
    results/final_results/tables/lmc_cross_model_results.csv \
    results/final_results/tables/lmc_adaptive_attacker_results.csv \
    results/final_results/tables/lmc_search_history.csv \
    results/final_results/env/env_report.txt \
    results/final_results/reports/final_experiment_summary.md; do
    if [ -f "$f" ]; then
        echo "    [OK] $f"
    else
        echo "    [MISSING] $f"
    fi
done
