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

## Available Methods (2025)

| Method | Category | Status | Notes |
|--------|----------|--------|-------|
| qmsanet | Denoising | PASS | Deep learning denoising |
| kervrann | Change Detection | PASS | Requires `ipol_kervrann` conda env |
| cstrd | Detection | PASS | Requires `ipol_cstrd` conda env |
| phase_unwrap | Phase Processing | PASS | |
| nerf_specularity | 3D Reconstruction | PASS | Uses pre-rendered models |
| semiogram | Medical | PASS | Gait analysis from IMU data |
| gaussian_splatting | 3D Reconstruction | CUDA | Requires NVIDIA GPU |
| latent_diffusion | Generation | CUDA | Requires NVIDIA GPU |
| sign_lmsls | Segmentation | DOCKER | Requires IPOL Docker |
| sign_asslisu | Segmentation | DOCKER | Requires IPOL Docker |

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

## Roadmap

See [TODO.md](TODO.md) for planned expansion to earlier IPOL years (2024, 2023, 2022...).

## License

Individual IPOL methods retain their original licenses. See each method folder for details.
