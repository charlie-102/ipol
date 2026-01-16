# IPOL 2023 Methods - Complete Status

## Overview

**Total articles in IPOL 2023:** 17
**Dataset-only (excluded):** 2 (389, 497)
**Methods with code:** 15

**Adapter Status:**
- ✅ Implemented: 15/15 (ALL methods with code!)
- IDs: 356, 401, 414, 417, 418, 420, 440, 441, 443, 446, 447, 459, 460, 467, 477
- Note: 414 was MATLAB, now converted to Python using finufft

## Complete Article List

| # | ID | Title | Type | Downloaded | Adapter | Status |
|---|-----|-------|------|------------|---------|--------|
| 1 | 356 | Robust Homography Estimation from Local Affine Maps | Python/C++ | ✅ | ✅ | Needs libDA build |
| 2 | 389 | Fall Detection with Smart Floor Sensors | Dataset | - | - | **Excluded** |
| 3 | 401 | Binary Shape Vectorization by Affine Scale-space | C++ | ✅ | ✅ | **✅ BUILT** |
| 4 | 414 | EPR Image Reconstruction with Total Variation | Python | ✅ | ✅ | **✅ Ready** (converted from MATLAB) |
| 5 | 417 | Signal Decomposition (Jump, Oscillation, Trend) | Python | ✅ | ✅ | **✅ Ready** |
| 6 | 418 | Progressive Compression of Triangle Meshes | C++ | ✅ | ✅ | Needs CGAL/Draco build |
| 7 | 420 | Signal-dependent Video Noise Estimator | Python | ✅ | ✅ | Needs TIFF input |
| 8 | 440 | One-Shot Federated Learning | Python | ✅ | ✅ | **✅ Ready** (converted) |
| 9 | 441 | GANet Stereo Matching | Python | ✅ | ✅ | Needs model weights |
| 10 | 443 | Fast Chromatic Aberration Correction | Python/Cython | ✅ | ✅ | **✅ BUILT** |
| 11 | 446 | MPRNet Image Restoration | Python | ✅ | ✅ | **✅ Ready** (weights present) |
| 12 | 447 | Semantic Segmentation Zoo | Python | ✅ | ✅ | Needs MMSeg checkpoints |
| 13 | 459 | Monocular Depth Estimation (4 methods) | Python | ✅ | ✅ | Needs model weights |
| 14 | 460 | Handheld Burst Super-Resolution | Python | ✅ | ✅ | **✅ Ready** |
| 15 | 467 | BSDE Image Denoising | C++ | ✅ | ✅ | **✅ BUILT** |
| 16 | 477 | OpenCCO Vascular Tree Generation | C++ | ✅ | ✅ | Needs DGtal + Ceres |
| 17 | 497 | Gait Data Set with IMU | Dataset | - | - | **Excluded** |

## Download Links for Missing Methods

### 414 - EPR Image Reconstruction
```
https://www.ipol.im/pub/art/2023/414/ipol-demo-tvepr-master.zip
```
- **Type:** Python
- **Description:** Total Variation regularization for EPR image reconstruction

### 459 - Monocular Depth Estimation (4 methods)
```
https://www.ipol.im/pub/art/2023/459/MiDaS-main.zip
https://www.ipol.im/pub/art/2023/459/DPT-master.zip
https://www.ipol.im/pub/art/2023/459/Adabins-main.zip
https://www.ipol.im/pub/art/2023/459/GLPDepth-main.zip
```
- **Type:** Python (PyTorch)
- **Description:** Review of 4 monocular depth estimation methods

## Status by Category

### Ready to Use (3)
- 401 shape_vectorization - Binary at `build/main`
- 417 signal_decomposition - Python script ready
- 460 burst_superres - Dependencies available
- 467 bsde_denoising - Binary at `BSDE`

### Needs Build (4)
- 356 homography - `cmake && make` for libDA
- 420 video_noise - `python setup.py build_ext --inplace`
- 443 chromatic_aberration - `python setup.py build_ext --inplace`
- 477 opencco - Needs DGtal from source + Ceres

### Needs Model Weights (3)
- 441 ganet - Download from IPOL
- 446 mprnet - Download from IPOL
- 447 segmentation_zoo - Download checkpoints from IPOL

### Not Implemented (2)
- 418 mesh_compression - Complex CGAL/MEPP2/Draco dependencies
- 440 federated_learning - Jupyter notebook (not CLI-compatible)

### Not Downloaded (2)
- 414 epr_reconstruction - See download link above
- 459 monocular_depth - See download links above (4 methods)

### Excluded Datasets (2)
- 389 - Fall Detection Data Set
- 497 - Gait Data Set

## Build Instructions

### 401 - Shape Vectorization (COMPLETED)
```bash
cd methods/ipol_2023_401_shape_vectorization
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/libpng ..
make
# Binary: build/main
```

### 467 - BSDE Denoising (COMPLETED)
```bash
cd methods/ipol_2023_467_bsde_denoising
clang++ -std=c++11 -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include \
  -I/opt/homebrew/opt/libpng/include \
  -L/opt/homebrew/opt/libomp/lib \
  -L/opt/homebrew/opt/libpng/lib \
  -lomp -lpng -O3 -o BSDE BSDE.cpp libdenoising.cpp lib.cpp io_png.c
```

### 477 - OpenCCO (Requires DGtal)
```bash
# Install Ceres
brew install ceres-solver

# Build DGtal from source
git clone https://github.com/DGtal-team/DGtal.git
cd DGtal && mkdir build && cd build
cmake .. && make && sudo make install

# Then build OpenCCO
cd methods/ipol_2023_477_opencco
mkdir build && cd build
cmake .. && make
```

## Summary Statistics

| Category | Count | IDs |
|----------|-------|-----|
| Total 2023 articles | 17 | |
| Excluded (datasets) | 2 | 389, 497 |
| Methods with code | 15 | |
| Downloaded | 15 | |
| **Adapters created** | **15** | ALL! |
| Built & Ready | 8 | 401, 414, 417, 440, 443, 446, 460, 467 |
| Needs model weights | 3 | 441, 447, 459 |
| Needs build/deps | 4 | 356, 418, 420, 477 |
