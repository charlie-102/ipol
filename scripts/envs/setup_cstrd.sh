#!/bin/bash
# Setup conda environment for CS-TRD (needs shapely 1.7)
# Usage: bash setup_cstrd.sh

set -e

ENV_NAME="ipol_cstrd"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IPOL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Setting up $ENV_NAME environment ==="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found"
    exit 1
fi

# Remove if exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing $ENV_NAME..."
    conda env remove -n $ENV_NAME -y
fi

# Create environment with conda packages (avoids pip proxy issues)
echo "Creating conda environment with packages..."
conda create -n $ENV_NAME python=3.10 \
    numpy=1.26 \
    matplotlib \
    opencv \
    pandas \
    scikit-learn \
    pillow \
    imageio \
    -c conda-forge -y

# Install shapely 1.7 via conda
echo "Installing shapely 1.7..."
conda run -n $ENV_NAME conda install "shapely<2" -c conda-forge -y

# Install remaining packages
conda run -n $ENV_NAME conda install natsort glob2 -c conda-forge -y

# Compile Devernay edge detector
echo "Compiling Devernay edge detector..."
DEVERNAY_DIR="$IPOL_DIR/methods/ipol_2025_485_cstrd/externas/devernay_1.0"
if [[ -d "$DEVERNAY_DIR" ]]; then
    cd "$DEVERNAY_DIR"
    make
    cd -
fi

echo ""
echo "=== $ENV_NAME setup complete ==="
echo "To activate:  conda activate $ENV_NAME"
echo "To test:      conda activate $ENV_NAME && python -m ipol_runner test cstrd"
