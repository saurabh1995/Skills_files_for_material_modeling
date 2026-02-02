#!/usr/bin/env bash
# =============================================================================
# setup_wsl_ubuntu.sh
#
# Purpose:
#   Same goal as DGX script, but optimized for Windows + Ubuntu (WSL2).
#   Default is CPU-safe (works on any WSL install).
#
# What we install:
#   1) Conda env (isolated Python)
#   2) fenics-dolfinx (FEniCSx/DOLFINx) from conda-forge
#   3) JAX (CPU) via pip
#   4) matplotlib for plots
#   5) dolfinx_materials from GitHub
#
# How to run (inside Ubuntu / WSL terminal):
#   chmod +x setup_wsl_ubuntu.sh
#   ./setup_wsl_ubuntu.sh
#
# After install:
#   conda activate vp_damage_wsl
#   python Materil_model_test_VP_Damage.py
#
# Optional GPU in WSL:
#   - First ensure GPU works inside WSL:
#       nvidia-smi
#     If that works, you can replace:
#       pip install -U jax
#     with:
#       pip install -U "jax[cuda12]"
# =============================================================================

set -euo pipefail

ENV_NAME="vp_damage_wsl"
PY_VER="3.10"

echo "==> [WSL] Pre-flight checks..."

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in Ubuntu."
  echo "Install Miniconda/Mambaforge inside Ubuntu/WSL, then re-run."
  exit 1
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

echo "==> Installing JAX (CPU-safe default)..."
python -m pip install -U jax

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
echo
echo "Optional WSL GPU:"
echo "  - If 'nvidia-smi' works inside Ubuntu, change JAX install line to:"
echo "      pip install -U \"jax[cuda12]\""
echo "================================================================="
