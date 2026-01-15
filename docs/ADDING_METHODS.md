# Adding IPOL Methods - Guide

## Lessons Learned from 2025 Methods

### Common Issues & Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| **shapely 2.x incompatible** | `LineString.__new__() takes from 1 to 2 positional arguments` | Use conda env with `shapely<2` |
| **numba/numpy version** | `Numba needs NumPy 2.3 or less` | Use conda env with `numpy<2.4` |
| **iio package unavailable** | `module 'iio' has no attribute 'read'` | Use `iio_shim.py` wrapper around imageio |
| **CUDA required** | `Torch not compiled with CUDA enabled` | Mark `requires_cuda = True`, skip on Mac |
| **Docker paths hardcoded** | `FileNotFoundError: /workdir/bin/...` | Mark `requires_docker = True` |
| **Model dtype mismatch** | `mat1 and mat2 must have the same dtype` | Ensure sample data uses float32 |
| **Sample data format** | `IndexError: list index out of range` | Check dataloader for expected file naming |

### Method Categories

```python
class MethodCategory(Enum):
    DENOISING = "denoising"
    CHANGE_DETECTION = "change_detection"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    GENERATION = "generation"
    RECONSTRUCTION_3D = "3d_reconstruction"
    PHASE_PROCESSING = "phase_processing"
    MEDICAL = "medical"
```

### Input Types

```python
class InputType(Enum):
    IMAGE = "image"
    IMAGE_PAIR = "image_pair"
    VIDEO = "video"
    POSE_DATA = "pose_data"
    SENSOR_DATA = "sensor_data"
    DATASET_ID = "dataset_id"
```

---

## Steps to Add a New Method

### 1. Download Method from IPOL

```bash
# Browse https://www.ipol.im/ for the method
# Download and extract to methods/ipol_YEAR_ID_name/
```

### 2. Create Adapter

Create `ipol_runner/methods/<method_name>.py`:

```python
"""Method Name adapter (IPOL XXX)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class MethodNameMethod(IPOLMethod):
    """Brief description."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_YEAR_ID_name"

    @property
    def name(self) -> str:
        return "method_name"

    @property
    def display_name(self) -> str:
        return "Method Display Name"

    @property
    def description(self) -> str:
        return "What the method does"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION  # Choose appropriate

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE  # Choose appropriate

    @property
    def input_count(self) -> int:
        return 1  # Number of input files

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    # Add if method needs NVIDIA GPU
    # @property
    # def requires_cuda(self) -> bool:
    #     return True

    # Add if method needs IPOL Docker
    # @property
    # def requires_docker(self) -> bool:
    #     return True

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "param1": {
                "type": "float",  # float, int, str, bool, choice
                "default": 1.0,
                "description": "Parameter description"
            },
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        input_path = inputs[0]

        cmd = [
            sys.executable, str(self.METHOD_DIR / "main.py"),
            "--input", str(input_path),
            "--output", str(output_dir),
            # Add other args
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check for outputs
            output_file = output_dir / "output.png"
            if output_file.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_file,
                    outputs={"output": output_file}
                )

            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"No output. stderr: {result.stderr}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
```

### 3. Import in `__init__.py`

Add to `ipol_runner/methods/__init__.py`:

```python
from . import method_name
```

### 4. Add Sample Data

Create test samples in `samples/<method_name>/` or `samples/<input_type>/`

### 5. Test

```bash
python -m ipol_runner test method_name
```

### 6. If Dependency Conflicts

Create conda env setup script in `scripts/envs/setup_<method_name>.sh`

---

## 2025 Methods Summary

| Method | Category | Status | Notes |
|--------|----------|--------|-------|
| qmsanet | DENOISING | ✓ PASS | Deep learning denoising |
| kervrann | CHANGE_DETECTION | ✓ PASS | Needs ipol_kervrann env (numba + iio shim) |
| cstrd | DETECTION | ✓ PASS | Needs ipol_cstrd env (shapely 1.7) |
| phase_unwrap | PHASE_PROCESSING | ✓ PASS | |
| nerf_specularity | RECONSTRUCTION_3D | ✓ PASS | Uses pre-rendered models |
| semiogram | MEDICAL | ✓ PASS | Gait analysis from IMU |
| gaussian_splatting | RECONSTRUCTION_3D | ⊘ CUDA | Needs NVIDIA GPU |
| latent_diffusion | GENERATION | ⊘ CUDA | Needs NVIDIA GPU |
| sign_lmsls | SEGMENTATION | ⊘ DOCKER | Needs IPOL Docker |
| sign_asslisu | SEGMENTATION | ⊘ DOCKER | Needs IPOL Docker |

---

## Environment Setup Scripts

Located in `scripts/envs/`:

- `setup_cstrd.sh` - shapely<2 for tree ring detection
- `setup_kervrann.sh` - numpy<2.4 + iio shim for change detection
- `iio_shim.py` - imageio wrapper providing iio API
