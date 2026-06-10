#!/usr/bin/env bash
# =============================================================================
# RunPod setup for SAFE-OpenVLA LIBERO rollout collection.
#
# Translates the working Colab flow in SAFE_OpenVLA.ipynb to a RunPod pod:
#   1. apt packages for headless MuJoCo/EGL rendering
#   2. Miniconda env (python 3.10) -- replaces condacolab
#   3. torch 2.2.0 cu121 first, then the pinned dependency stack
#   4. flash-attn 2.5.5 (--no-build-isolation, after torch is in place)
#   5. editable installs of openvla / LIBERO / dlimp with --no-deps so the
#      dependency solver can't drift the fragile NumPy/TF pins
#   6. ~/.libero/config.yaml patch (LIBERO otherwise demands a /datasets dir)
#   7. import smoke test (numpy / tensorflow / torch+cuda / flash_attn / libero)
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
CONDA_DIR="${CONDA_DIR:-$ROOT_DIR/miniconda3}"
ENV_NAME="safe-openvla"

echo ">>> openvla repo: $OPENVLA_DIR"
echo ">>> install root: $ROOT_DIR"

# --- 1. System packages (headless EGL rendering for MuJoCo/robosuite) --------
apt-get update -qq
apt-get install -y -qq \
    libegl1 libgl1 libglfw3 libglew2.2 patchelf ffmpeg \
    git wget ninja-build

# --- 2. Clone sibling repos ---------------------------------------------------
[ -d "$LIBERO_DIR" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
[ -d "$DLIMP_DIR" ]  || git clone https://github.com/kvablack/dlimp "$DLIMP_DIR"

# --- 3. Miniconda + python 3.10 env (stack is pinned against 3.10) ------------
if [ ! -d "$CONDA_DIR" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV_NAME " || conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME"

# --- 4. PyTorch first (cu121, exact pins from requirements.txt) ---------------
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# --- 5. Pinned dependency stack (general + LIBERO extras, like the Colab) -----
pip install \
    accelerate draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 \
    tokenizers==0.19.1 transformers==4.40.1 wandb \
    tensorflow==2.15.0 tensorflow-datasets==4.9.3 tensorflow-graphics==2021.12.3 \
    pandas scikit-learn "imageio[ffmpeg]" robosuite==1.4.0 bddl easydict \
    cloudpickle gym bitsandbytes

# --- 6. Flash Attention 2 (after torch; never with build isolation) -----------
pip install packaging ninja
ninja --version
pip cache remove flash_attn 2>/dev/null || true
pip install "flash-attn==2.5.5" --no-build-isolation

# --- 7. Editable installs WITHOUT dependency solving --------------------------
# (lesson from Colab: letting pip resolve these drifts the NumPy/TF pins)
pip install --no-deps -e "$OPENVLA_DIR"
pip install --no-deps -e "$LIBERO_DIR"
pip install --no-deps -e "$DLIMP_DIR"

# Repair NumPy in case anything dragged in numpy>=2 (TF 2.15 requires <2)
pip install "numpy==1.26.4"

# --- 8. LIBERO config patch ----------------------------------------------------
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

# --- 9. Environment file to source before collecting data ---------------------
cat > "$OPENVLA_DIR/env_runpod.sh" <<EOF
# source this before running rollout collection
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MPLBACKEND=Agg
export PRISMATIC_DATA_ROOT=$ROOT_DIR
export HF_HOME=$ROOT_DIR/hf_cache   # keep the 7B checkpoint on the persistent volume
EOF

# --- 10. Smoke test -------------------------------------------------------------
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
