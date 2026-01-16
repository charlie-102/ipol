# IPOL Runner

Unified CLI for running and comparing [IPOL](https://www.ipol.im/) (Image Processing On Line) methods.

## Features

- Run IPOL methods with a consistent CLI interface
- Compare outputs from multiple methods side-by-side
- Generate visual galleries for result comparison
- Track method validation status
- Support for CUDA and Docker-dependent methods

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ipol

# Install base dependencies
pip install numpy matplotlib pillow imageio

# Install method-specific dependencies
python -m ipol_runner deps <method_name>
```

## Quick Start

```bash
# List available methods
python -m ipol_runner list

# Get method info
python -m ipol_runner info qmsanet

# Run a method
python -m ipol_runner run qmsanet -i noisy_image.png -o ./output

# Test all methods
python -m ipol_runner test

# Show validation status
python -m ipol_runner status
```

## Available Methods

### By Category

| Category | Methods | Input Type |
|----------|---------|------------|
| **Denoising** | qmsanet, dark_channel | image |
| **Change Detection** | kervrann | image_pair |
| **Detection** | cstrd, noisesniffer, line_segment, tsne, slavc | image/dataset_id |
| **Segmentation** | sign_lmsls, sign_asslisu, interactive_seg, domain_seg | pose_data/image |
| **Generation** | latent_diffusion, bigcolor, icolorit, image_abstraction, superpixel_color | image/image_pair |
| **3D Reconstruction** | gaussian_splatting, nerf_specularity, nerf_vaxnerf, storm | image/dataset_id |
| **Phase Processing** | phase_unwrap, phinet | image/image_pair |
| **Medical** | semiogram, armcoda, survival_forest | sensor_data |

### IPOL 2025 Methods

| Method | Category | Input | Status | Notes |
|--------|----------|-------|--------|-------|
| qmsanet | Denoising | image | PASS | Deep learning denoising |
| kervrann | Change Detection | image_pair | PASS | Requires `ipol_kervrann` conda env |
| cstrd | Detection | image | PASS | Requires `ipol_cstrd` conda env |
| phase_unwrap | Phase Processing | image | PASS | Delaunay triangulation |
| nerf_specularity | 3D Reconstruction | dataset_id | PASS | Pre-rendered models |
| semiogram | Medical | sensor_data | PASS | Gait analysis from IMU |
| gaussian_splatting | 3D Reconstruction | image | CUDA | Requires NVIDIA GPU |
| latent_diffusion | Generation | image | CUDA | Requires NVIDIA GPU |
| sign_lmsls | Segmentation | pose_data | DOCKER | Requires IPOL Docker |
| sign_asslisu | Segmentation | pose_data | DOCKER | Requires IPOL Docker |

### IPOL 2024 Methods

| Method | Category | Input | Status | Notes |
|--------|----------|-------|--------|-------|
| dark_channel | Denoising | image | PASS | Python/C++ backends |
| image_abstraction | Generation | image | PASS | Python/C++ backends |
| noisesniffer | Detection | image | PASS | Noise-based forgery detection |
| line_segment | Detection | image | PASS | 8 algorithm comparison |
| tsne | Detection | dataset_id | PASS | t-SNE visualization |
| armcoda | Medical | sensor_data | PASS | Motion capture analysis |
| storm | 3D Reconstruction | image | PASS | Super-resolution microscopy |
| phinet | Phase Processing | image_pair | SPECIAL | InSAR data (.npy), PyTorch |
| survival_forest | Medical | sensor_data | SPECIAL | Built-in datasets |
| bigcolor | Generation | image | CUDA | BigGAN colorization |
| icolorit | Generation | image | CUDA | Interactive colorization |
| nerf_vaxnerf | 3D Reconstruction | dataset_id | CUDA | Visual hull NeRF |
| superpixel_color | Generation | image_pair | CUDA | VGG-based color transfer |
| interactive_seg | Segmentation | image | CUDA | Click-based segmentation |
| domain_seg | Segmentation | image | CUDA | Domain generalization |
| slavc | Detection | image+audio | CUDA | Sound localization |

## CLI Commands

```
python -m ipol_runner <command> [options]

Commands:
  list              List available methods
  info <method>     Show method details and parameters
  run <method>      Run a method on input(s)
  test [method]     Test method(s) with sample data
  status            Show validation status
  deps <method>     Install method dependencies
  compare           Compare outputs from multiple methods
  gallery           Generate web gallery for visual comparison
```

### Examples

```bash
# Run QMSANet denoising with specific model
python -m ipol_runner run qmsanet -i image.png --param model=S50

# Run Kervrann change detection (requires two images)
python -m ipol_runner run kervrann -i before.tif -i after.tif --param scale=3

# Run CS-TRD tree ring detection (requires center coordinates)
python -m ipol_runner run cstrd -i tree.png --param cx=500 --param cy=500

# Compare multiple methods
python -m ipol_runner compare -m qmsanet method2 -i noisy.png -o ./comparison

# Generate visual gallery
python -m ipol_runner gallery -o ./output --open
```

## Environment Setup for Specific Methods

Some methods require isolated conda environments due to dependency conflicts.

### CS-TRD (Tree Ring Detection)

Requires shapely < 2 for compatibility:

```bash
bash scripts/envs/setup_cstrd.sh
conda activate ipol_cstrd
python -m ipol_runner test cstrd
```

### Kervrann (Change Detection)

Requires numpy < 2.4 for numba + iio shim:

```bash
bash scripts/envs/setup_kervrann.sh
conda activate ipol_kervrann
python -m ipol_runner test kervrann
```

## Adding New Methods

See [docs/ADDING_METHODS.md](docs/ADDING_METHODS.md) for:
- Step-by-step guide
- Common issues and solutions
- Method adapter template

## Project Structure

```
ipol/
├── ipol_runner/              # CLI package
│   ├── __main__.py           # Entry point
│   ├── cli.py                # CLI commands
│   ├── base.py               # Base classes
│   ├── registry.py           # Method registry
│   ├── runner.py             # Execution logic
│   ├── testing.py            # Test framework
│   ├── validation.py         # Status tracking
│   └── methods/              # Method adapters
├── methods/                  # Downloaded IPOL methods
├── samples/                  # Test input data
├── scripts/envs/             # Conda env setup scripts
├── docs/                     # Documentation
└── TODO.md                   # Roadmap
```

## Backend Conversions

Some methods have been converted to pure Python for easier deployment:

| Method | Original | Converted | Switch |
|--------|----------|-----------|--------|
| phinet | TensorFlow/Keras | PyTorch | PyTorch only |
| dark_channel | C++ | Python | `--param backend=python` (default) or `cpp` |
| image_abstraction | C++/Qt4 | Python | `--param backend=python` (default) or `cpp` |

## Roadmap

See [TODO.md](TODO.md) for planned expansion to earlier IPOL years (2024, 2023, 2022...).

## License

Individual IPOL methods retain their original licenses. See each method folder for details.
