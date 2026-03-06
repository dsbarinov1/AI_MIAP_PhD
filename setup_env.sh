#!/usr/bin/env bash
# Setup MIAP PhD environment: install Miniconda if missing, create conda env,
# install ecole (and SCIP) via conda, install the rest via pip.
# Usage: bash setup_env.sh [env_name]
# Run from project root. Then: conda activate <env_name> (default: miap_phd)

set -e
ENV_NAME="${1:-miap_phd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
if [ -n "$CONDA_PREFIX" ]; then
  CONDA_DIR="$(dirname "$(dirname "$CONDA_PREFIX")")"
fi

# --- Find or install Conda ---
# Check common install locations if conda not in PATH
if ! command -v conda &>/dev/null; then
  for d in "$HOME/miniconda3" "$HOME/Miniconda3" "$HOME/anaconda3" "$HOME/Anaconda3" \
           "$(echo "$USERPROFILE" 2>/dev/null | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/\1|')/miniconda3" \
           "$(echo "$USERPROFILE" 2>/dev/null | sed 's|\\|/|g' | sed 's|^\([A-Za-z]\):|/\1|')/Miniconda3"; do
    [ -z "$d" ] && continue
    if [ -x "$d/bin/conda" ] 2>/dev/null; then
      CONDA_DIR="$d"
      . "$d/bin/activate" 2>/dev/null || true
      export PATH="$d/bin:$d/Scripts:$PATH"
      break
    fi
    if [ -x "$d/Scripts/conda.exe" ] 2>/dev/null; then
      CONDA_DIR="$d"
      export PATH="$d/Scripts:$d/Library/bin:$PATH"
      break
    fi
  done
fi

if ! command -v conda &>/dev/null; then
  echo "[setup_env] conda not found, installing Miniconda..."
  UNAME="$(uname -s)"
  ARCH="$(uname -m)"
  case "$UNAME" in
    Linux)
      if [ "$ARCH" = "x86_64" ]; then
        INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
      else
        INSTALLER="Miniconda3-latest-Linux-${ARCH}.sh"
      fi
      DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"
      if [ ! -f "$INSTALLER" ]; then
        echo "[setup_env] Downloading $DOWNLOAD_URL"
        (command -v wget &>/dev/null && wget -q "$DOWNLOAD_URL" -O "$INSTALLER") || \
        (command -v curl &>/dev/null && curl -sL "$DOWNLOAD_URL" -o "$INSTALLER") || \
        { echo "[setup_env] Need wget or curl."; exit 1; }
      fi
      bash "$INSTALLER" -b -p "$CONDA_DIR"
      rm -f "$INSTALLER"
      . "$CONDA_DIR/bin/activate"
      ;;
    Darwin)
      if [ "$ARCH" = "arm64" ]; then
        INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
      else
        INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
      fi
      DOWNLOAD_URL="https://repo.anaconda.com/miniconda/${INSTALLER}"
      if [ ! -f "$INSTALLER" ]; then
        echo "[setup_env] Downloading $DOWNLOAD_URL"
        (command -v wget &>/dev/null && wget -q "$DOWNLOAD_URL" -O "$INSTALLER") || \
        (command -v curl &>/dev/null && curl -sL "$DOWNLOAD_URL" -o "$INSTALLER") || \
        { echo "[setup_env] Need wget or curl."; exit 1; }
      fi
      bash "$INSTALLER" -b -p "$CONDA_DIR"
      rm -f "$INSTALLER"
      . "$CONDA_DIR/bin/activate"
      ;;
    MINGW64_NT-*|MSYS_NT-*|CYGWIN_NT-*)
      echo "[setup_env] Windows (Git Bash) detected. Install Miniconda manually:"
      echo "  1. Download: https://docs.conda.io/en/latest/miniconda.html (Windows x86_64)"
      echo "  2. Run the installer and add Miniconda to PATH (or use Anaconda Prompt)"
      echo "  3. Open a new terminal, cd to this project, run: bash setup_env.sh"
      echo "  Or in Anaconda Prompt: conda create -n $ENV_NAME python=3.10 && conda activate $ENV_NAME && conda install -c conda-forge ecole && pip install -r requirements.txt"
      exit 1
      ;;
    *)
      echo "[setup_env] Unsupported OS: $UNAME. Install conda manually: conda create -n $ENV_NAME python=3.10; conda activate $ENV_NAME; conda install -c conda-forge ecole; pip install -r requirements.txt"
      exit 1
      ;;
  esac
  echo "[setup_env] Miniconda installed to $CONDA_DIR"
  conda init bash 2>/dev/null || true
else
  echo "[setup_env] Using existing conda: $(conda --version)"
fi

# Ensure we have conda in path (e.g. after fresh install)
if ! command -v conda &>/dev/null; then
  # shellcheck source=/dev/null
  . "$CONDA_DIR/bin/activate"
fi

# --- Create env if missing ---
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "[setup_env] Conda env '$ENV_NAME' already exists. Activate it and run: pip install -r requirements.txt"
else
  echo "[setup_env] Creating conda env '$ENV_NAME' with python=3.10..."
  conda create -n "$ENV_NAME" python=3.10 -y
fi

# --- Activate and install: ecole (conda) + rest (pip) ---
echo "[setup_env] Activating '$ENV_NAME' and installing packages..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[setup_env] Installing ecole (and SCIP) via conda..."
conda install -c conda-forge ecole -y

echo "[setup_env] Installing pip requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[setup_env] Verifying imports..."
python -c "
import ecole
import torch
import torch_geometric
import pyscipopt
import numpy
print('ecole:', ecole.__version__ if hasattr(ecole, \"__version__\") else 'ok')
print('torch:', torch.__version__)
print('pyscipopt: ok')
print('All deps OK.')
"

echo ""
echo "[setup_env] Done. To activate later: conda activate $ENV_NAME"
echo "[setup_env] Then run: python data_collector.py ... && python train.py ..."
