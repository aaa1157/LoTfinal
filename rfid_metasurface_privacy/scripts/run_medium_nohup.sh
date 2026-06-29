#!/usr/bin/env bash
# run_medium_nohup.sh - 后台运行 medium 实验
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PID_FILE="results/logs/medium_run.pid"

# 检查是否已有实验在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Medium experiment already running (PID: $OLD_PID)"
        echo "  To check: bash scripts/check_status.sh"
        echo "  To kill:  bash scripts/kill_experiment.sh medium"
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

# 创建日志目录
mkdir -p results/logs

# 生成日志文件名
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/logs/medium_run_${TIMESTAMP}.log"

# 创建后台运行脚本
RUN_SCRIPT="results/logs/medium_run_${TIMESTAMP}.sh"
cat > "$RUN_SCRIPT" << 'RUNEOF'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "[1/5] Building dataset (medium, scene_disjoint) ..."
python build_dataset.py --mode medium --split scene_disjoint --force
echo ""

echo "[2/5] Running main.py (medium) ..."
python main.py --mode medium
echo ""

echo "[3/5] Training deep attacker (medium, seeds 2026 2027 2028) ..."
python train_deep_attacker.py --mode medium --seeds 2026 2027 2028
echo ""

echo "[4/5] Training learnable controller (medium) ..."
python train_learnable_controller.py --mode medium
echo ""

echo "[5/5] Generating report assets ..."
python generate_report_assets.py
echo ""

echo "MEDIUM experiment complete at $(date)"
# 清理 PID 文件
rm -f results/logs/medium_run.pid
RUNEOF

chmod +x "$RUN_SCRIPT"

# 写入 PID 文件标记
echo $$ > "$PID_FILE"

# 使用 nohup 后台运行
nohup bash "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &
REAL_PID=$!

# 更新 PID 为实际进程
echo $REAL_PID > "$PID_FILE"

echo "============================================================"
echo "  Medium experiment started in background."
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
echo "    bash scripts/kill_experiment.sh medium"
echo ""
