# Adding IPOL Methods - Comprehensive Guide

## Quick Reference

### Method Types Encountered

| Type | Examples | Key Patterns |
|------|----------|--------------|
| **Pure Python** | qmsanet, tsne | Direct subprocess call |
| **C++ with Python wrapper** | dark_channel, image_abstraction | Compile or use Python fallback |
| **Legacy TensorFlow** | phinet | Convert to PyTorch |
| **PyTorch models** | bigcolor, icolorit | Need CUDA, download weights |
| **Jupyter/Quarto** | survival_forest | Use built-in datasets or convert to script |
| **Multi-algorithm** | line_segment | Wrapper with algorithm selection |
| **Docker-dependent** | sign_lmsls | Mark requires_docker, can't run locally |

---

## Step 1: Explore the IPOL Method

### 1.1 Download and Extract

```bash
cd methods
# From IPOL article page, download source code
curl -LO https://www.ipol.im/pub/art/YEAR/ID/ID-main.zip
unzip ID-main.zip
mv ID-main ipol_YEAR_ID_shortname/
```

### 1.2 Find the Entry Point

Look for these files in order:
1. `main.py` - Most common
2. `run.py` or `demo.py` - Alternative names
3. `test.py` - Sometimes the actual runner
4. `*.sh` scripts - Check what Python script they call
5. `DDL.json` - IPOL demo definition, lists the command

```bash
# Quick way to find entry points
ls *.py *.sh
cat DDL.json 2>/dev/null | head -50
head -50 main.py
```

### 1.3 Identify Parameters

```bash
# Check argparse arguments
grep -n "add_argument" main.py
grep -n "argparse" *.py

# Check DDL.json for parameter definitions
cat DDL.json | grep -A5 '"param"'
```

### 1.4 Check Dependencies

```bash
cat requirements.txt
cat setup.py 2>/dev/null | head -30
cat pyproject.toml 2>/dev/null

# Key things to look for:
# - tensorflow/keras -> Consider PyTorch conversion
# - CUDA/torch -> Mark requires_cuda
# - Qt/GUI libs -> May need headless alternative
# - R dependencies -> Complex setup
```

### 1.5 Understand Input/Output

```bash
# Check what files it reads/writes
grep -n "imread\|read\|load" main.py
grep -n "imwrite\|save\|write" main.py

# Check for hardcoded paths (Docker methods)
grep -n "/workdir\|/tmp" *.py
```

---

## Step 2: Classify the Method

### Categories

| Category | Description | Example Methods |
|----------|-------------|-----------------|
| `DENOISING` | Remove noise, dehaze | qmsanet, dark_channel |
| `CHANGE_DETECTION` | Compare two images | kervrann |
| `DETECTION` | Find features/objects | cstrd, noisesniffer, line_segment |
| `SEGMENTATION` | Segment regions | sign_lmsls, interactive_seg |
| `GENERATION` | Create/modify images | bigcolor, image_abstraction |
| `RECONSTRUCTION_3D` | 3D from 2D | gaussian_splatting, storm |
| `PHASE_PROCESSING` | Phase unwrap/denoise | phase_unwrap, phinet |
| `MEDICAL` | Clinical analysis | semiogram, survival_forest |

### Input Types

| Input Type | Description | Sample Location |
|------------|-------------|-----------------|
| `IMAGE` | Single image | `samples/image/` |
| `IMAGE_PAIR` | Two images | `samples/image_pair/` |
| `VIDEO` | Video file | `samples/video/` |
| `POSE_DATA` | Skeleton/pose files | `samples/pose_data/` |
| `SENSOR_DATA` | IMU/tabular data | `samples/sensor_data/` |
| `DATASET_ID` | Predefined dataset | Built into method |

---

## Step 3: Handle Special Cases

### 3.1 Converting TensorFlow/Keras to PyTorch

**When to convert**: Legacy TensorFlow 1.x, Keras with custom layers

**Pattern used for phinet**:

```python
# 1. Extract architecture from Keras HDF5
import h5py
with h5py.File('model.h5', 'r') as f:
    config = f.attrs.get('model_config')
    # Parse JSON config to understand layers

# 2. Recreate in PyTorch
class ModelPyTorch(nn.Module):
    def __init__(self):
        # Match Keras layer structure
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.bn1 = nn.BatchNorm2d(out_ch)
        ...

# 3. Convert weights
def convert_keras_to_pytorch(keras_path, pytorch_path):
    with h5py.File(keras_path, 'r') as f:
        weights = f['model_weights']

        # Keras Conv2D: (H, W, C_in, C_out)
        # PyTorch Conv2d: (C_out, C_in, H, W)
        kernel = np.transpose(kernel, (3, 2, 0, 1))

        # ConvTranspose2d: swap first two dims
        if is_transpose_conv:
            kernel = np.transpose(kernel, (1, 0, 2, 3))
```

**Key gotchas**:
- BatchNorm has `running_mean`, `running_var` (not just weight/bias)
- ConvTranspose2d weight dims are (in, out, H, W) not (out, in, H, W)
- Keras uses NHWC, PyTorch uses NCHW

### 3.2 Converting C++ to Python (with Backend Switch)

**When to do this**: C++ requires compilation, Qt dependencies, platform-specific

**Pattern used for dark_channel, image_abstraction**:

```python
@register
class MethodName(IPOLMethod):

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "backend": {
                "type": "choice",
                "choices": ["python", "cpp"],
                "default": "python",
                "description": "Python (portable) or C++ (faster, requires compilation)"
            },
            # ... other params
        }

    def run(self, inputs, output_dir, params):
        backend = params.get("backend", "python")

        if backend == "cpp":
            return self._run_cpp(inputs[0], output_dir, params)
        else:
            return self._run_python(inputs[0], output_dir, params)

    def _run_python(self, input_path, output_dir, params):
        # Import pure Python implementation
        from .method_python import process_image
        result = process_image(input_path, **params)
        ...

    def _run_cpp(self, input_path, output_dir, params):
        # Try to compile if needed
        compiled, error = self._ensure_compiled()
        if not compiled:
            return MethodResult(success=False, error_message=error)

        # Run binary
        cmd = [str(self.METHOD_DIR / "binary"), ...]
        ...
```

**Python implementation tips**:
- Use scipy.ndimage for filters (minimum_filter, gaussian_filter, uniform_filter)
- Use numpy for array operations
- Can be simplified approximation if full algorithm is complex

### 3.3 Methods with Built-in Datasets

**When applicable**: survival_forest, nerf_specularity, tsne

```python
def get_parameters(self):
    return {
        "dataset": {
            "type": "choice",
            "choices": ["builtin1", "builtin2", "custom"],
            "default": "builtin1",
            "description": "Use built-in dataset or provide custom input"
        }
    }

def run(self, inputs, output_dir, params):
    dataset = params.get("dataset", "builtin1")

    if dataset == "custom" and inputs:
        data_path = inputs[0]
    else:
        data_path = self._get_builtin_dataset(dataset)
```

### 3.4 Multi-Input Methods

**Examples**: kervrann (image_pair), phinet (complex InSAR pair), slavc (image + audio)

```python
@property
def input_type(self) -> InputType:
    return InputType.IMAGE_PAIR  # or IMAGE for mixed inputs

@property
def input_count(self) -> int:
    return 2

def run(self, inputs, output_dir, params):
    if len(inputs) < 2:
        return MethodResult(
            success=False,
            error_message="Requires 2 inputs: image1 and image2"
        )

    input1 = inputs[0]
    input2 = inputs[1]
```

### 3.5 Methods Requiring Model Weights

**Pattern for downloading weights**:

```python
WEIGHTS_URL = "https://example.com/weights.pth"

def _ensure_weights(self) -> tuple[bool, str]:
    weights_path = self.METHOD_DIR / "weights" / "model.pth"

    if weights_path.exists():
        return True, ""

    weights_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request
        urllib.request.urlretrieve(self.WEIGHTS_URL, str(weights_path))
        return True, ""
    except Exception as e:
        return False, f"Failed to download weights: {e}"
```

---

## Step 4: Common Issues & Solutions

### Dependency Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| shapely 2.x | `LineString.__new__() takes 1-2 args` | `conda install shapely<2` |
| numpy 2.4+ | `Numba needs NumPy 2.3 or less` | `conda install numpy<2.4` |
| TensorFlow 1.x | `ModuleNotFoundError: tensorflow.contrib` | Convert to PyTorch |
| Qt4 required | `qmake: command not found` | Use Python backend |
| CUDA required | `Torch not compiled with CUDA` | Mark `requires_cuda = True` |
| iio package | `module 'iio' has no attribute 'read'` | Use iio_shim.py |

### Runtime Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Docker paths | `FileNotFoundError: /workdir/...` | Mark `requires_docker = True` |
| Dtype mismatch | `mat1 and mat2 must have same dtype` | Convert to float32 |
| Shape mismatch | `RuntimeError: size mismatch` | Check weight conversion |
| No output | Method runs but no files created | Check output path in method |
| Timeout | Hangs on large images | Add timeout, resize input |

### Weight Conversion Issues

| Issue | Solution |
|-------|----------|
| Conv2D dims wrong | Transpose: `(3, 2, 0, 1)` for Keras→PyTorch |
| ConvTranspose2d dims | Additional transpose: `(1, 0, 2, 3)` |
| BatchNorm missing keys | Include running_mean, running_var |
| Layer names don't match | Create explicit mapping dict |

---

## Step 5: Testing Checklist

```bash
# 1. Verify registration
python -m ipol_runner list | grep method_name

# 2. Check info displays correctly
python -m ipol_runner info method_name

# 3. Install dependencies
python -m ipol_runner deps method_name

# 4. Run test
python -m ipol_runner test method_name

# 5. Check validation status
python -m ipol_runner status

# 6. If CUDA required, verify skip message
# [method_name] SKIP - Requires NVIDIA CUDA GPU
```

---

## Adapter Templates

### Simple Image Method

```python
"""Method Name adapter (IPOL YEAR article ID)."""
from pathlib import Path
from typing import Any, Dict, List
from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register

@register
class MethodNameMethod(IPOLMethod):
    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_YEAR_ID_name"

    @property
    def name(self) -> str:
        return "method_name"

    @property
    def display_name(self) -> str:
        return "Method Display Name"

    @property
    def description(self) -> str:
        return "What it does"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def run(self, inputs: List[Path], output_dir: Path, params: Dict[str, Any]) -> MethodResult:
        try:
            import numpy as np
            import imageio.v2 as imageio

            image = imageio.imread(str(inputs[0]))
            # Process image...
            result = image  # Your processing here

            output_path = output_dir / "result.png"
            imageio.imwrite(str(output_path), result.astype(np.uint8))

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=output_path,
                outputs={"result": output_path}
            )
        except Exception as e:
            return MethodResult(success=False, output_dir=output_dir, error_message=str(e))
```

### Dual Backend Method (Python + C++)

```python
@register
class MethodNameMethod(IPOLMethod):
    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_YEAR_ID_name"

    @property
    def requires_compilation(self) -> bool:
        return False  # Python backend doesn't need compilation

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "backend": {
                "type": "choice",
                "choices": ["python", "cpp"],
                "default": "python",
                "description": "Backend: python (portable) or cpp (faster)"
            }
        }

    def run(self, inputs, output_dir, params):
        backend = params.get("backend", "python")
        if backend == "cpp":
            return self._run_cpp(inputs[0], output_dir, params)
        return self._run_python(inputs[0], output_dir, params)

    def _run_python(self, input_path, output_dir, params):
        from .method_python import process
        # ... implementation

    def _run_cpp(self, input_path, output_dir, params):
        # ... compile and run binary
```

### CUDA-Required Method

```python
@register
class MethodNameMethod(IPOLMethod):
    @property
    def requires_cuda(self) -> bool:
        return True

    def run(self, inputs, output_dir, params):
        try:
            import torch
            if not torch.cuda.is_available():
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message="CUDA not available"
                )
            # ... rest of implementation
```

---

## 2024 Methods Summary

| Method | Type | Conversion | Status |
|--------|------|------------|--------|
| dark_channel | C++ | Python backend added | ✓ PASS |
| image_abstraction | C++/Qt4 | Python backend added | ✓ PASS |
| phinet | TensorFlow/Keras | Converted to PyTorch | ✓ PASS (needs InSAR data) |
| noisesniffer | Pure Python | None needed | ✓ PASS |
| line_segment | Multi-algorithm | None needed | ✓ PASS |
| tsne | Pure Python | None needed | ✓ PASS |
| storm | Pure Python | None needed | ✓ PASS |
| armcoda | Pure Python | None needed | ✓ PASS |
| survival_forest | Jupyter | Built-in datasets | ✓ PASS |
| bigcolor | PyTorch | None needed | ⊘ CUDA |
| icolorit | PyTorch | None needed | ⊘ CUDA |
| nerf_vaxnerf | JAX | None needed | ⊘ CUDA |
| superpixel_color | PyTorch | None needed | ⊘ CUDA |
| interactive_seg | PyTorch | None needed | ⊘ CUDA |
| domain_seg | PyTorch | None needed | ⊘ CUDA |
| slavc | PyTorch | None needed | ⊘ CUDA |

---

## File Naming Convention

```
methods/
  ipol_YEAR_ID_shortname/     # Source code from IPOL

ipol_runner/methods/
  shortname.py                # Main adapter
  shortname_python.py         # Python backend (if converted from C++)
  shortname_pytorch.py        # PyTorch model (if converted from TF)
```

---

## Environment Setup Scripts

For methods with conflicting dependencies, create `scripts/envs/setup_<method>.sh`:

```bash
#!/bin/bash
conda create -n ipol_methodname python=3.10 -y
conda activate ipol_methodname
pip install -r requirements.txt
# Additional fixes...
```
