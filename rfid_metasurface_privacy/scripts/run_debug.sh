#!/usr/bin/env bash
# run_debug.sh - 前台快速检查
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 激活虚拟环境
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "============================================================"
echo "  RFID Metasurface Privacy - DEBUG Mode (Foreground)"
echo "============================================================"
echo "  Project: $PROJECT_DIR"
echo "  Time: $(date)"
echo ""

echo "[1/5] Building dataset (debug) ..."
python build_dataset.py --mode debug --force
echo ""

echo "[2/5] Running main.py (debug) ..."
python main.py --mode debug
echo ""

echo "[3/5] Training deep attacker (debug) ..."
python train_deep_attacker.py --mode debug
echo ""

echo "[4/5] Training learnable controller (debug) ..."
python train_learnable_controller.py --mode debug
echo ""

echo "[5/5] Generating report assets ..."
python generate_report_assets.py
echo ""

echo "============================================================"
echo "  DEBUG run complete!"
echo "============================================================"
