#!/usr/bin/env bash
# =============================================================================
# RunPod setup for SAFE-OpenVLA LIBERO rollout collection.
#
# Translates the working Colab flow in SAFE_OpenVLA.ipynb to a RunPod pod,
# installing straight into the pod's base python (no conda/venv -- RunPod
# images work best that way):
#   1. apt packages for headless MuJoCo/EGL rendering
#   2. torch 2.2.0 cu121 first, then the pinned dependency stack
#   3. flash-attn 2.5.5 (--no-build-isolation, after torch is in place)
#   4. editable installs of openvla / LIBERO / dlimp with --no-deps so the
#      dependency solver can't drift the fragile NumPy/TF pins
#   5. ~/.libero/config.yaml patch (LIBERO otherwise demands a /datasets dir)
#   6. import smoke test (numpy / tensorflow / torch+cuda / flash_attn / libero)
#
# Usage (from a fresh pod):
#   cd /workspace && git clone https://github.com/WaterPancake/openvla
#   bash /workspace/openvla/setup_runpod.sh
#
# Afterwards, before collecting data:
#   source /workspace/openvla/env_runpod.sh
# =============================================================================
set -euo pipefail

# --- Paths -------------------------------------------------------------------
# Keep everything on the persistent volume so it survives pod restarts.
ROOT_DIR="${ROOT_DIR:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENVLA_DIR="$SCRIPT_DIR"
LIBERO_DIR="$ROOT_DIR/LIBERO"
DLIMP_DIR="$ROOT_DIR/dlimp"

echo ">>> openvla repo: $OPENVLA_DIR"
echo ">>> install root: $ROOT_DIR"
echo ">>> python: $(which python) ($(python --version 2>&1))"
# Stack is pinned against python 3.10; 3.11 usually works too. Warn otherwise.
python - <<'PY'
import sys
if sys.version_info[:2] not in [(3, 10), (3, 11)]:
    print(f"WARNING: python {sys.version.split()[0]} -- the pinned stack "
          "(torch 2.2.0 / tf 2.15 / tokenizers 0.19.1) expects 3.10 or 3.11. "
          "Pick a RunPod image with python 3.10 if installs fail.")
PY

# --- 1. System packages (headless EGL rendering for MuJoCo/robosuite) --------
apt-get update -qq
apt-get install -y -qq \
    libegl1 libgl1 libglfw3 libglew2.2 patchelf ffmpeg \
    git wget ninja-build

# --- 2. Clone sibling repos ---------------------------------------------------
[ -d "$LIBERO_DIR" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
[ -d "$DLIMP_DIR" ]  || git clone https://github.com/kvablack/dlimp "$DLIMP_DIR"

# --- 3. PyTorch first (cu121, exact pins from requirements.txt) ---------------
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# --- 4. Pinned dependency stack (general + LIBERO extras, like the Colab) -----
pip install \
    accelerate draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 \
    tokenizers==0.19.1 transformers==4.40.1 wandb \
    tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics==2021.12.3 \
    pandas scikit-learn "imageio[ffmpeg]" robosuite==1.4.0 bddl easydict \
    cloudpickle gym bitsandbytes

# --- 5. Flash Attention 2 (after torch) ----------------------------------------
# Use the prebuilt wheel from the GitHub release -- no nvcc compile (~minutes
# saved). Built against cu122/torch2.2, which is the official pairing for any
# torch 2.2.x cu12x install (flash-attn's own setup.py maps cu121 -> cu122 wheel).
pip install packaging ninja
CPTAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-${CPTAG}-${CPTAG}-linux_x86_64.whl"
pip install "$FA_WHEEL" || {
    echo ">>> Prebuilt wheel unavailable for $CPTAG; falling back to source build."
    pip cache remove flash_attn 2>/dev/null || true
    pip install "flash-attn==2.5.5" --no-build-isolation
}

# --- 6. Editable installs WITHOUT dependency solving --------------------------
# (lesson from Colab: letting pip resolve these drifts the NumPy/TF pins)
pip install --no-deps -e "$OPENVLA_DIR"
pip install --no-deps -e "$LIBERO_DIR"
pip install --no-deps -e "$DLIMP_DIR"

# Repair NumPy in case anything dragged in numpy>=2 (TF 2.15 requires <2)
pip install "numpy==1.26.4"

# --- 7. LIBERO config patch ----------------------------------------------------
# LIBERO insists on a datasets dir at first import; point everything at the clone.
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<YAML
benchmark_root: $LIBERO_DIR/libero/libero
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
init_states: $LIBERO_DIR/libero/libero/init_files
datasets: $LIBERO_DIR/libero/datasets
assets: $LIBERO_DIR/libero/libero/assets
YAML
touch "$LIBERO_DIR/libero/__init__.py"

# --- 8. Environment file to source before collecting data ---------------------
cat > "$OPENVLA_DIR/env_runpod.sh" <<EOF
# source this before running rollout collection
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export PRISMATIC_DATA_ROOT=$ROOT_DIR
export HF_HOME=$ROOT_DIR/hf_cache   # keep the 7B checkpoint on the persistent volume
EOF

# --- 9. Smoke test -------------------------------------------------------------
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg
python - <<'PY'
import numpy as np;        print("numpy:", np.__version__)
import tensorflow as tf;   print("tensorflow:", tf.__version__)
import torch;              print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
import flash_attn;         print("flash-attn:", flash_attn.__version__)
from libero.libero import benchmark
print("LIBERO suites:", list(benchmark.get_benchmark_dict().keys()))
PY

echo
echo "=============================================================="
echo "Setup complete. Before collecting rollouts, run:"
echo "  source $OPENVLA_DIR/env_runpod.sh"
echo
echo "Then e.g.:"
echo "  cd $OPENVLA_DIR"
echo "  python experiments/robot/libero/run_libero_eval.py \\"
echo "    --model_family openvla \\"
echo "    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-10 \\"
echo "    --task_suite_name libero_10 \\"
echo "    --num_trials_per_task 50 \\"
echo "    --center_crop True \\"
echo "    --output_hidden_states True \\"
echo "    --attn_implementation flash_attention_2 \\"
echo "    --use_wandb False --save_logs True --seed 10 --resume True \\"
echo "    --save_root $ROOT_DIR/openvla_rollouts"
echo "=============================================================="
