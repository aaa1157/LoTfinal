#!/bin/bash
# 收集实验环境信息
# 输出: results/final_results/env/env_report.txt 和 env_report.json

set -e

OUT_DIR="results/final_results/env"
mkdir -p "$OUT_DIR"

TXT="$OUT_DIR/env_report.txt"
JSON="$OUT_DIR/env_report.json"

echo "Collecting environment info..."

# Helper: write to both txt and collect json pairs
> "$TXT"

echo "=== Experiment Environment Report ===" >> "$TXT"
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TXT"
echo "" >> "$TXT"

# 1. OS
OS_INFO=$(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo "Unknown")
echo "OS: $OS_INFO" >> "$TXT"

# 2. User
USER_NAME=$(whoami)
echo "User: $USER_NAME" >> "$TXT"

# 3. Project path
PROJ_PATH=$(pwd)
echo "Project Path: $PROJ_PATH" >> "$TXT"

# 4. Python version
PY_VER=$(python --version 2>&1 | awk '{print $2}')
echo "Python: $PY_VER" >> "$TXT"

# 5. pip version
PIP_VER=$(pip --version 2>&1 | awk '{print $2}')
echo "pip: $PIP_VER" >> "$TXT"

# 6. Package versions
NUMPY_VER=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "N/A")
SCIPY_VER=$(python -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "N/A")
PANDAS_VER=$(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "N/A")
SKLEARN_VER=$(python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo "N/A")
MPL_VER=$(python -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null || echo "N/A")
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")

echo "numpy: $NUMPY_VER" >> "$TXT"
echo "scipy: $SCIPY_VER" >> "$TXT"
echo "pandas: $PANDAS_VER" >> "$TXT"
echo "scikit-learn: $SKLEARN_VER" >> "$TXT"
echo "matplotlib: $MPL_VER" >> "$TXT"
echo "torch: $TORCH_VER" >> "$TXT"

# 7-8. PyTorch / CUDA
TORCH_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda or 'N/A')" 2>/dev/null || echo "N/A")
CUDNN_VER=$(python -c "import torch; print(torch.backends.cudnn.version() or 'N/A')" 2>/dev/null || echo "N/A")

echo "PyTorch available: $TORCH_AVAIL" >> "$TXT"
echo "CUDA available: $CUDA_AVAIL" >> "$TXT"
echo "torch.version.cuda: $TORCH_CUDA_VER" >> "$TXT"
echo "cuDNN version: $CUDNN_VER" >> "$TXT"

# 11-12. GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
echo "GPU: $GPU_NAME" >> "$TXT"
echo "GPU Memory: $GPU_MEM" >> "$TXT"

# 14-16. CPU / Memory
CPU_MODEL=$(lscpu 2>/dev/null | grep "Model name" | awk -F: '{$1=""; print substr($0,2)}' || echo "N/A")
CPU_CORES=$(nproc 2>/dev/null || echo "N/A")
MEM_TOTAL=$(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo "N/A")
echo "CPU: $CPU_MODEL" >> "$TXT"
echo "CPU Cores: $CPU_CORES" >> "$TXT"
echo "Memory: $MEM_TOTAL" >> "$TXT"

# 17. Disk
DISK_FREE=$(df -h . 2>/dev/null | awk 'NR==2{print $4}' || echo "N/A")
echo "Disk Free: $DISK_FREE" >> "$TXT"

# 18. Git
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "N/A")
echo "Git Commit: $GIT_COMMIT" >> "$TXT"

# 13. nvidia-smi full
echo "" >> "$TXT"
echo "=== nvidia-smi ===" >> "$TXT"
nvidia-smi >> "$TXT" 2>/dev/null || echo "nvidia-smi not available" >> "$TXT"

echo "" >> "$TXT"
echo "=== Environment collection complete ===" >> "$TXT"

# Generate JSON
python -c "
import json
data = {
    'os': '$OS_INFO',
    'user': '$USER_NAME',
    'project_path': '$PROJ_PATH',
    'python': '$PY_VER',
    'pip': '$PIP_VER',
    'numpy': '$NUMPY_VER',
    'scipy': '$SCIPY_VER',
    'pandas': '$PANDAS_VER',
    'scikit_learn': '$SKLEARN_VER',
    'matplotlib': '$MPL_VER',
    'torch': '$TORCH_VER',
    'torch_available': '$TORCH_AVAIL' == 'True',
    'cuda_available': '$CUDA_AVAIL' == 'True',
    'torch_cuda_version': '$TORCH_CUDA_VER',
    'cudnn_version': '$CUDNN_VER',
    'gpu_name': '$GPU_NAME',
    'gpu_memory': '$GPU_MEM',
    'cpu_model': '$CPU_MODEL',
    'cpu_cores': '$CPU_CORES',
    'memory_total': '$MEM_TOTAL',
    'disk_free': '$DISK_FREE',
    'git_commit': '$GIT_COMMIT',
}
with open('$JSON', 'w') as f:
    json.dump(data, f, indent=2)
print('JSON saved')
"

echo "Environment info saved to $TXT and $JSON"
