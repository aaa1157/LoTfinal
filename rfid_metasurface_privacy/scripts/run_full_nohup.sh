#!/usr/bin/env bash
# run_full_nohup.sh - 后台运行 full 实验
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PID_FILE="results/logs/full_run.pid"

# 检查是否已有实验在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Full experiment already running (PID: $OLD_PID)"
        echo "  To check: bash scripts/check_status.sh"
        echo "  To kill:  bash scripts/kill_experiment.sh full"
        exit 1
    else
        echo "Stale PID file found, removing..."
        rm -f "$PID_FILE"
    fi
fi

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

mkdir -p results/logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/full_run_${TIMESTAMP}.log"

RUN_SCRIPT="results/logs/full_run_${TIMESTAMP}.sh"
cat > "$RUN_SCRIPT" << 'RUNEOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "[1/5] Building dataset (full, scene_disjoint) ..."
python build_dataset.py --mode full --split scene_disjoint --force
echo ""

echo "[2/5] Running main.py (full) ..."
python main.py --mode full
echo ""

echo "[3/5] Training deep attacker (full, seeds 2026 2027 2028) ..."
python train_deep_attacker.py --mode full --seeds 2026 2027 2028
echo ""

echo "[4/5] Training learnable controller (full) ..."
python train_learnable_controller.py --mode full
echo ""

echo "[5/5] Generating report assets ..."
python generate_report_assets.py
echo ""

echo "FULL experiment complete at $(date)"
rm -f results/logs/full_run.pid
RUNEOF

chmod +x "$RUN_SCRIPT"

echo $$ > "$PID_FILE"

nohup bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &
REAL_PID=$!

echo $REAL_PID > "$PID_FILE"

echo "============================================================"
echo "  Full experiment started in background."
echo "============================================================"
echo ""
echo "  PID:  $REAL_PID"
echo "  Log:  $LOG_FILE"
echo ""
echo "  View log:"
echo "    tail -f $LOG_FILE"
echo ""
echo "  Check process:"
echo "    ps -p $REAL_PID -f"
echo ""
echo "  Check status:"
echo "    bash scripts/check_status.sh"
echo ""
echo "  Stop experiment:"
echo "    bash scripts/kill_experiment.sh full"
echo ""
