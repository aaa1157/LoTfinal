#!/bin/bash
# Kill final experiment
PID_FILE="results/final_results/logs/final_medium.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Killing process $PID..."
    kill "$PID" 2>/dev/null || echo "Process not found"
    # Also kill any child python processes
    pkill -f "train_deep_attacker.py" 2>/dev/null || true
    pkill -f "train_learnable_controller.py" 2>/dev/null || true
    pkill -f "run_lmc_experiments.py" 2>/dev/null || true
    pkill -f "main.py" 2>/dev/null || true
    echo "Done."
else
    echo "No PID file found. Trying to kill by name..."
    pkill -f "train_deep_attacker.py" 2>/dev/null || true
    pkill -f "train_learnable_controller.py" 2>/dev/null || true
    pkill -f "run_lmc_experiments.py" 2>/dev/null || true
    pkill -f "main.py" 2>/dev/null || true
    echo "Done."
fi
