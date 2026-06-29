#!/bin/bash
# Kill overnight pipeline

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PID_FILE="results/final_results/logs/overnight.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Nothing to kill."
    exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
    echo "Killing pipeline (PID=$PID)..."
    kill "$PID" 2>/dev/null
    sleep 2
    if kill -0 "$PID" 2>/dev/null; then
        echo "Force killing..."
        kill -9 "$PID" 2>/dev/null
    fi
    echo "Pipeline killed."
else
    echo "PID=$PID is not running."
fi

rm -f "$PID_FILE"
