#!/usr/bin/env bash
set -e

echo "============================================"
echo "  RFID Metasurface Privacy Simulation"
echo "  Environment Setup Script"
echo "============================================"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ---- 1. Check python3 ----
echo ""
echo "[1/8] Checking python3 ..."
if command -v python3 &>/dev/null; then
    PYTHON_VER=$(python3 --version 2>&1)
    echo "  OK: $PYTHON_VER"
else
    echo "  ERROR: python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# ---- 2. Create .venv ----
echo ""
echo "[2/8] Creating virtual environment .venv ..."
if [ -d ".venv" ]; then
    echo "  .venv already exists, skipping."
else
    python3 -m venv .venv
    echo "  .venv created."
fi

# ---- 3. Activate .venv ----
echo ""
echo "[3/8] Activating .venv ..."
source .venv/bin/activate
echo "  Active Python: $(which python3)"

# ---- 4. Upgrade pip ----
echo ""
echo "[4/8] Upgrading pip ..."
python3 -m pip install --upgrade pip --quiet 2>&1 | tail -1
echo "  pip version: $(python3 -m pip --version 2>&1 | awk '{print $2}')"

# ---- 5. Install requirements.txt ----
echo ""
echo "[5/8] Installing requirements.txt ..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt --quiet 2>&1 | tail -3
    echo "  Requirements installed."
else
    echo "  WARNING: requirements.txt not found, skipping."
fi

# ---- 6. Check core scientific packages ----
echo ""
echo "[6/8] Checking core scientific packages ..."
for pkg in numpy scipy matplotlib pandas sklearn tqdm; do
    python3 -c "import ${pkg}" 2>/dev/null && echo "  OK: ${pkg}" || echo "  MISSING: ${pkg}"
done

# ---- 7. Check torch ----
echo ""
echo "[7/8] Checking torch ..."
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
    echo "  OK: torch $TORCH_VER"
else
    echo "  torch not available in .venv."
    echo "  To install: pip install torch (CPU only) or follow https://pytorch.org"
    echo "  Basic simulation (main.py) will still run fine without torch."
fi

# ---- 8. Check CUDA ----
echo ""
echo "[8/8] Checking CUDA ..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_DEV=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  CUDA available: YES ($CUDA_DEV)"
else
    echo "  CUDA available: NO (deep learning will use CPU)"
fi

# ---- Done ----
echo ""
echo "============================================"
echo "  Environment setup complete!"
echo "============================================"
echo ""
echo "  Run the following commands to start:"
echo ""
echo "    source .venv/bin/activate"
echo "    python main.py"
echo "    python train_deep_attacker.py --fast"
echo "    python train_learnable_controller.py --fast"
echo ""
