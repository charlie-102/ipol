#!/bin/bash
# Download model weights for IPOL methods
# These weights are required for certain methods to function properly

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=================================="
echo "IPOL Method Model Weight Downloader"
echo "=================================="
echo ""
echo "Base directory: $BASE_DIR"
echo ""

# Function to download a file
download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ]; then
        echo "✓ $desc already exists"
        return 0
    fi

    echo "Downloading $desc..."
    mkdir -p "$(dirname "$dest")"
    curl -L -o "$dest" "$url"
    if [ -f "$dest" ]; then
        echo "✓ Downloaded $desc"
    else
        echo "✗ Failed to download $desc"
        return 1
    fi
}

# Monocular Depth (MiDaS)
echo ""
echo "=== Monocular Depth (MiDaS) ==="
MIDAS_DIR="$BASE_DIR/methods/ipol_2023_459_monocular_depth/MiDaS-main/weights"
mkdir -p "$MIDAS_DIR"
download_file \
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small.pt" \
    "$MIDAS_DIR/midas_v21_small.pt" \
    "MiDaS v2.1 Small weights"

# Segmentation Zoo (mmsegmentation checkpoints)
echo ""
echo "=== Segmentation Zoo (mmsegmentation) ==="
SEG_ZOO_DIR="$BASE_DIR/methods/ipol_2023_447_segmentation_zoo/checkpoints"
mkdir -p "$SEG_ZOO_DIR"
# Note: mmsegmentation checkpoints are hosted on OpenMMLab servers
echo "Download mmsegmentation checkpoints from:"
echo "  https://download.openmmlab.com/mmsegmentation/"
echo ""
echo "Required checkpoint: fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pth"
echo "Place in: $SEG_ZOO_DIR/"
echo ""
# Try direct download
download_file \
    "https://download.openmmlab.com/mmsegmentation/v0.5/fcn/fcn_r101-d8_512x512_160k_ade20k/fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pth" \
    "$SEG_ZOO_DIR/fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pth" \
    "FCN-R101 ADE20k checkpoint"

# GANet (Stereo Matching)
echo ""
echo "=== GANet (Stereo Matching) ==="
GANET_DIR="$BASE_DIR/methods/ipol_2023_441_ganet/models"
mkdir -p "$GANET_DIR"
echo "GANet requires CUDA and custom extensions."
echo "Download model weights from IPOL article 441:"
echo "  https://www.ipol.im/pub/art/2023/441/"
echo ""
echo "Place in: $GANET_DIR/sceneflow_epoch_10.pth"

# Homography (DL models)
echo ""
echo "=== Homography Estimation ==="
HOMO_DIR="$BASE_DIR/methods/ipol_2023_356_homography/DL/model_weights"
mkdir -p "$HOMO_DIR"
echo "Download from IPOL article 356:"
echo "  https://www.ipol.im/pub/art/2023/356/"
echo ""
echo "Required models:"
echo "  - HomographyNet_COCO.pth"
echo "  - HomographyNet_new_COCO.pth"
echo "  - HomographyNet_new_seq_COCO.pth"
echo "  - HomographyNet_seq_COCO.pth"

# MPRNet
echo ""
echo "=== MPRNet (Image Restoration) ==="
MPRNET_DIR="$BASE_DIR/methods/ipol_2023_446_mprnet/models"
mkdir -p "$MPRNET_DIR"
echo "Download from IPOL article 446:"
echo "  https://www.ipol.im/pub/art/2023/446/"
echo ""
echo "Required models:"
echo "  - model_denoising.pth"
echo "  - model_deblurring.pth"
echo "  - model_deraining.pth"

echo ""
echo "=================================="
echo "Download script complete!"
echo ""
echo "To verify which weights are present, run:"
echo "  python -m ipol_runner status"
echo "=================================="
