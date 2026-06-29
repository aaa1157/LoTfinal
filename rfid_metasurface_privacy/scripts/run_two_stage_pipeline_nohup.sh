#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="results/final_results/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/two_stage_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/two_stage.pid"

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Killing existing pipeline (PID=$OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
    fi
fi

nohup bash scripts/run_two_stage_pipeline.sh > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "============================================================"
echo "  Two-Stage Pipeline Launched!"
echo "============================================================"
echo "  PID:        $(cat $PID_FILE)"
echo "  Log:        $LOG_FILE"
echo "  Monitor:    tail -f $LOG_FILE"
echo "  Status:     bash scripts/check_two_stage_status.sh"
echo "  Kill:       bash scripts/kill_two_stage_pipeline.sh"
echo "============================================================"
