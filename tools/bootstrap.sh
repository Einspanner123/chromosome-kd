#!/bin/bash
# LDMDet 一键环境部署
# Usage: bash tools/bootstrap.sh
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
step() { echo -e "${GREEN}>>> $1${NC}"; }
err()  { echo -e "${RED}!!! $1${NC}"; exit 1; }

# ── 1. 创建 conda 环境 ──────────────────────────
ENV_NAME="${1:-ldmdet}"
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

# ── 3. mmcv + mmengine (match PyTorch version) ───
step "installing mmcv + mmengine"
pip install mmcv==2.1.0 mmengine==0.10.5

# ── 4. 项目本身 (mmdet + ldmdet, editable) ──────
step "installing project (mmdet + ldmdet)"
pip install -e .

# ── 5. 额外依赖 ──────────────────────────────────
step "installing extras"
pip install pycocotools timm opencv-python swanlab

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
for n in ['random','hard_ot','sinkhorn_stochastic','ghss']: build_coupling(n, epsilon=5.0)
print('ldmdet OK')
"

step "done. run: bash tools/train.sh experiments/configs/ldmdet/rf_heun_adaln.py --seed 42"
