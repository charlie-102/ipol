#!/bin/bash
# Setup conda environment for Kervrann (needs numpy < 2.4 for numba)
# Usage: bash setup_kervrann.sh

set -e

ENV_NAME="ipol_kervrann"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# numpy < 2.4 for numba compatibility
echo "Creating conda environment with packages..."
conda create -n $ENV_NAME python=3.10 \
    "numpy<2.4" \
    numba \
    matplotlib \
    scipy \
    scikit-image \
    pillow \
    -c conda-forge -y

# Install imageio (for iio shim)
echo "Installing imageio..."
conda run -n $ENV_NAME conda install imageio -c conda-forge -y

# Install iio shim (wraps imageio to provide iio-compatible API)
echo "Installing iio shim..."
SITE_PACKAGES=$(conda run -n $ENV_NAME python -c "import site; print(site.getsitepackages()[0])")
cp "$SCRIPT_DIR/iio_shim.py" "$SITE_PACKAGES/iio.py"

# Verify iio installation
echo "Verifying iio shim..."
conda run -n $ENV_NAME python -c "import iio; print('iio version:', iio.version); print('iio.read exists:', hasattr(iio, 'read'))"

echo ""
echo "=== $ENV_NAME setup complete ==="
echo "To activate:  conda activate $ENV_NAME"
echo "To test:      conda activate $ENV_NAME && python -m ipol_runner test kervrann"
