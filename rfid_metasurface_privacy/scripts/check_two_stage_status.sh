#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PID_FILE="results/final_results/logs/two_stage.pid"
if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found."
    exit 0
fi
PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Pipeline is RUNNING (PID=$PID)"
else
    echo "Pipeline is NOT RUNNING (PID=$PID may have finished)"
fi
LOG_DIR="results/final_results/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/two_stage_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo ""
    echo "Latest log: $LATEST_LOG"
    echo "Last 30 lines:"
    echo "---"
    tail -30 "$LATEST_LOG"
fi
