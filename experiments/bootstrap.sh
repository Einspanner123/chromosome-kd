#!/bin/bash
# KaryoFlow 环境部署
# Usage: bash experiments/bootstrap.sh [env_name]
set -euo pipefail

# 切到项目根目录 (experiments/ -> 项目根)
cd "$(dirname "$0")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
step() { echo -e "${GREEN}>>> $1${NC}"; }
err()  { echo -e "${RED}ERROR: $1${NC}"; exit 1; }

# ── 1. 创建 conda 环境 ──────────────────────────
ENV_NAME="${1:-karyoflow}"
if conda info --envs | grep -q "^${ENV_NAME} "; then
    step "conda env '${ENV_NAME}' exists, reusing"
else
    step "creating conda env: ${ENV_NAME} (Python 3.10)"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ── 2. PyTorch (auto-detect CUDA version) ────────
step "installing PyTorch"
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' 2>/dev/null || echo "0")
case "${CUDA_VER%%.*}" in
    12|13) TORCH_CUDA="cu121" ;;
    11)    TORCH_CUDA="cu118" ;;
    *)     TORCH_CUDA="cpu" ;;
esac
pip install torch==2.1.0 torchvision==0.16.0 --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# ── 3. OpenMMLab runtime (match PyTorch/CUDA) ───
step "installing OpenMMLab runtime"
pip install openmim
mim install mmcv==2.1.0
pip install mmengine==0.10.5 mmdet==3.3.0

# ── 4. KaryoFlow (editable) ──────
step "installing KaryoFlow"
pip install -e . --no-deps

# ── 5. 额外依赖 ──────────────────────────────────
step "installing extras"
pip install einops matplotlib numpy opencv-python pycocotools scipy swanlab

# ── 6. SwanLab 登录 ──────────────────────────────
step "SwanLab setup"
if [ -f ~/.swanlab/apikey ]; then
    echo "  key found"
elif [ -n "${SWANLAB_API_KEY:-}" ]; then
    python -c "import swanlab; swanlab.login(api_key='${SWANLAB_API_KEY}')"
else
    echo "  skip (set SWANLAB_API_KEY env var or run: swanlab login)"
fi

# ── 7. 验证 ──────────────────────────────────────
step "verifying"
python -c "
import torch; assert torch.cuda.is_available(), 'CUDA missing'
print(f'PyTorch {torch.__version__} + CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
import mmdet, mmengine, mmcv; print(f'mmdet {mmdet.__version__} | mmengine {mmengine.__version__} | mmcv {mmcv.__version__}')
from ldmdet.coupling import build_coupling
build_coupling('random')
build_coupling('ot_flow', epsilon=5.0, num_iters=20)
print('ldmdet OK')
"

step "done. inspect a run with: python tools/experiments/launch.py --help"
