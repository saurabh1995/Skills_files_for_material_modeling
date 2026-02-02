#!/usr/bin/env bash
# =============================================================================
# setup_dgx_gpu.sh
#
# Purpose:
#   Create a clean conda environment and install everything needed to run:
#     - Viscoplasticity_with_Damage_KH.py
#     - Materil_model_test_VP_Damage.py
#     - (and other scripts that import dolfinx_materials / dolfinx)
#
# What we install:
#   1) Conda env (isolated Python)
#   2) fenics-dolfinx (FEniCSx/DOLFINx) from conda-forge
#      - brings MPI, PETSc, UFL/Basix/FFCx, etc. (compiled stack)
#   3) JAX with CUDA support (GPU) via pip
#   4) matplotlib for plots
#   5) dolfinx_materials from GitHub
#
# How to run (on DGX, inside Linux shell):
#   chmod +x setup_dgx_gpu.sh
#   ./setup_dgx_gpu.sh
#
# After install:
#   conda activate vp_damage_dgx
#   python Materil_model_test_VP_Damage.py
#
# Notes:
#   - Requires NVIDIA driver installed (nvidia-smi should work).
#   - Uses JAX "cuda12" wheels. If your DGX image is older/newer, you may
#     need to adjust the JAX install line below.
# =============================================================================

set -euo pipefail

ENV_NAME="vp_damage_dgx"
PY_VER="3.10"

echo "==> [DGX] Pre-flight checks..."

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found."
  echo "Install Miniconda/Mambaforge first (recommended on DGX), then re-run."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found."
  echo "JAX will likely run on CPU. Ensure NVIDIA drivers are installed."
else
  echo "==> NVIDIA GPU detected:"
  nvidia-smi || true
fi

# Make conda activation work in scripts
eval "$(conda shell.bash hook)"

echo "==> Creating env '${ENV_NAME}' (if it doesn't exist)..."
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "==> Setting safer conda defaults..."
conda config --env --set channel_priority strict >/dev/null 2>&1 || true

echo "==> Installing FEniCSx/DOLFINx (fenics-dolfinx) from conda-forge..."
conda install -y -c conda-forge fenics-dolfinx

echo "==> Upgrading pip..."
python -m pip install --upgrade pip

echo "==> Installing plotting + utilities..."
python -m pip install -U matplotlib

echo "==> Installing JAX (GPU, CUDA 12 wheels)..."
# If your DGX image needs a different CUDA wheel, adjust this line.
python -m pip install -U "jax[cuda12]"

echo "==> Installing dolfinx_materials from GitHub..."
python -m pip install -U "git+https://github.com/bleyerj/dolfinx_materials.git"

echo "==> Smoke checks (fail fast if something is wrong)..."
python - << 'PY'
import sys
print("Python:", sys.version)

# JAX check
import jax
print("JAX devices:", jax.devices())

# DOLFINx check
import dolfinx
print("dolfinx version:", dolfinx.__version__)

print("Smoke check OK.")
PY

echo
echo "================================================================="
echo "✅ Installation finished."
echo
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  python Materil_model_test_VP_Damage.py"
echo "================================================================="
