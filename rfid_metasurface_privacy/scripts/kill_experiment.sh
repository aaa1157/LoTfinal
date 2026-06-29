#!/usr/bin/env bash
# kill_experiment.sh - 停止后台实验
# Usage: bash scripts/kill_experiment.sh medium
#        bash scripts/kill_experiment.sh full

if [ -z "$1" ]; then
    echo "Usage: bash scripts/kill_experiment.sh [medium|full]"
    exit 1
fi

MODE=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PID_FILE="results/logs/${MODE}_run.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found for ${MODE} experiment."
    echo "  Expected: $PID_FILE"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process $PID is not running. Cleaning up PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Killing ${MODE} experiment (PID: $PID) ..."
kill "$PID"

# 等待 3 秒
sleep 3

if ps -p "$PID" > /dev/null 2>&1; then
    echo "WARNING: Process $PID did not stop gracefully."
    echo "  You can force kill with: kill -9 $PID"
else
    echo "Process $PID stopped successfully."
    rm -f "$PID_FILE"
fi
