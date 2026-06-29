#!/usr/bin/env bash
# check_status.sh - 查看后台实验状态

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  Experiment Status Check"
echo "============================================================"

FOUND_RUNNING=0

# 检查 medium
if [ -f "results/logs/medium_run.pid" ]; then
    PID=$(cat results/logs/medium_run.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo ""
        echo "  [RUNNING] Medium experiment (PID: $PID)"
        FOUND_RUNNING=1
    else
        echo ""
        echo "  [STOPPED] Medium experiment (stale PID: $PID)"
    fi
else
    echo ""
    echo "  [NONE] No medium experiment PID file."
fi

# 检查 full
if [ -f "results/logs/full_run.pid" ]; then
    PID=$(cat results/logs/full_run.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        echo ""
        echo "  [RUNNING] Full experiment (PID: $PID)"
        FOUND_RUNNING=1
    else
        echo ""
        echo "  [STOPPED] Full experiment (stale PID: $PID)"
    fi
else
    echo ""
    echo "  [NONE] No full experiment PID file."
fi

# 找最近的日志文件
echo ""
echo "------------------------------------------------------------"
echo "  Recent log files:"
echo "------------------------------------------------------------"
LATEST_LOG=""
for pattern in "medium_run" "full_run"; do
    if ls results/logs/${pattern}_*.log 1>/dev/null 2>&1; then
        LATEST=$(ls -t results/logs/${pattern}_*.log 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo "  $LATEST"
            if [ -z "$LATEST_LOG" ]; then
                LATEST_LOG="$LATEST"
            fi
        fi
    fi
done

if [ -z "$LATEST_LOG" ]; then
    echo "  No log files found."
fi

# 显示最近日志尾部
if [ -n "$LATEST_LOG" ] && [ -f "$LATEST_LOG" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "  Last 40 lines of: $LATEST_LOG"
    echo "------------------------------------------------------------"
    tail -n 40 "$LATEST_LOG"
fi

echo ""
if [ "$FOUND_RUNNING" -eq 0 ]; then
    echo "  No background experiments currently running."
fi
echo ""
