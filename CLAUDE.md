# IPOL Reproduction Project

This repository provides a unified CLI to run and compare image processing methods from [IPOL (Image Processing On Line)](https://www.ipol.im/).

## Project Structure

```
ipol/
├── CLAUDE.md              # This file - guidelines for reproduction
├── requirements.txt       # Common dependencies
├── ipol_runner/           # Unified CLI package
│   ├── __init__.py
│   ├── __main__.py        # Entry: python -m ipol_runner
│   ├── cli.py             # CLI implementation
│   ├── base.py            # Base classes (IPOLMethod, MethodCategory, InputType)
│   ├── registry.py        # Method registry
│   ├── runner.py          # Execution logic
│   ├── comparison.py      # Side-by-side comparison
│   └── methods/           # Method adapters
│       ├── __init__.py    # Import all adapters here
│       └── *.py           # One adapter per method
└── methods/               # IPOL method source code
    └── ipol_<year>_<id>_<name>/
```

## Naming Convention

Method folders follow: `ipol_<year>_<id>_<short_name>/`

Examples:
- `ipol_2025_602_kervrann/` - Kervrann change detector (IPOL 2025, article 602)
- `ipol_2024_xxx_method/` - Some 2024 method
- `ipol_2025_560a_sign_lmsls/` - Use `a`, `b` suffix when one paper has multiple methods

## Adding Methods from a New Year

### Step 1: Download Articles

1. Go to `https://www.ipol.im/pub/art/<YEAR>/`
2. For each article, download the source code ZIP from the article page
3. Extract to `methods/ipol_<year>_<id>_<name>/`

Example download script:
```bash
# Download IPOL 2024 articles
cd methods
curl -LO https://www.ipol.im/pub/art/2024/<ID>/<archive>.zip
unzip <archive>.zip
mv <extracted_folder> ipol_2024_<id>_<name>
```

### Step 2: Explore the Method

Before creating an adapter, understand the method:

1. **Find entry point**: Look for `main.py`, `run.py`, `demo.py`, or check README
2. **Identify parameters**: Check argparse arguments in the main script
3. **Determine input/output**: What files does it read/write?
4. **Check dependencies**: Look at `requirements.txt` or `pyproject.toml`

### Step 3: Create Adapter

Create `ipol_runner/methods/<method_name>.py`:

```python
"""<Method Name> adapter (IPOL <YEAR> article <ID>)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class MyMethod(IPOLMethod):
    """Brief description of the method."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_<year>_<id>_<name>"

    @property
    def name(self) -> str:
        return "my_method"  # CLI name (lowercase, underscores)

    @property
    def display_name(self) -> str:
        return "My Method Display Name"

    @property
    def description(self) -> str:
        return "One-line description of what it does"

    @property
    def category(self) -> MethodCategory:
        # Choose from: DENOISING, CHANGE_DETECTION, DETECTION, SEGMENTATION,
        # GENERATION, RECONSTRUCTION_3D, PHASE_PROCESSING, MEDICAL
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        # Choose from: IMAGE, IMAGE_PAIR, VIDEO, POSE_DATA, SENSOR_DATA, DATASET_ID
        return InputType.IMAGE

    @property
    def input_count(self) -> int:
        return 1  # Number of input files required

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "param1": {
                "type": "float",  # float, int, str, bool, choice
                "default": 1.0,
                "description": "Parameter description"
            },
            "param2": {
                "type": "choice",
                "choices": ["option1", "option2"],
                "default": "option1",
                "description": "Choose an option"
            },
            "required_param": {
                "type": "int",
                "required": True,
                "description": "This param is required"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Execute the method."""
        input_path = inputs[0]

        # Build command to run the original script
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            "--input", str(input_path),
            "--output", str(output_dir),
            "--param1", str(params.get("param1", 1.0)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check for expected outputs
            outputs = {}
            primary = None

            output_file = output_dir / "output.png"
            if output_file.exists():
                outputs["result"] = output_file
                primary = output_file

            if not primary:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output. stderr: {result.stderr}"
                )

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=primary,
                outputs=outputs
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

### Step 4: Register the Adapter

Add import to `ipol_runner/methods/__init__.py`:

```python
from . import my_method  # Add this line
```

### Step 5: Test

```bash
# Verify registration
python -m ipol_runner list

# Check method info
python -m ipol_runner info my_method

# Install dependencies
python -m ipol_runner deps my_method

# Run the method
python -m ipol_runner run my_method -i input.png -o ./output
```

## Categories Reference

| Category | Description | Example Methods |
|----------|-------------|-----------------|
| `DENOISING` | Remove noise from images | QMSANet |
| `CHANGE_DETECTION` | Detect changes between images | Kervrann |
| `DETECTION` | Detect features/objects | CS-TRD (tree rings) |
| `SEGMENTATION` | Segment regions/objects | Sign language segmentation |
| `GENERATION` | Generate new images | Latent diffusion |
| `RECONSTRUCTION_3D` | 3D reconstruction | Gaussian splatting, NeRF |
| `PHASE_PROCESSING` | Phase unwrapping/processing | L1-norm phase unwrap |
| `MEDICAL` | Medical/clinical analysis | Semiogram (gait) |

## Input Types Reference

| Input Type | Description | Example |
|------------|-------------|---------|
| `IMAGE` | Single image file | Most methods |
| `IMAGE_PAIR` | Two images | Change detection |
| `VIDEO` | Video file | Video analysis |
| `POSE_DATA` | Skeleton/pose files | Sign language |
| `SENSOR_DATA` | Sensor readings | Gait analysis (IMU) |
| `DATASET_ID` | Predefined dataset | NeRF specularity |

## CLI Reference

```bash
# List all methods
python -m ipol_runner list

# Filter by category
python -m ipol_runner list -c denoising
python -m ipol_runner list -c 3d_reconstruction

# Filter by input type
python -m ipol_runner list -t image
python -m ipol_runner list -t image_pair

# Verbose list
python -m ipol_runner list -v

# Method info
python -m ipol_runner info <method>

# Install dependencies
python -m ipol_runner deps <method>

# Run method
python -m ipol_runner run <method> -i <input> -o <output>
python -m ipol_runner run <method> -i img1 -i img2 -o <output>  # For image pairs
python -m ipol_runner run <method> -i <input> --param key=value -o <output>

# Compare methods (same input type)
python -m ipol_runner compare -m method1 method2 -i <input> -o <output>

# Test methods with sample inputs
python -m ipol_runner test              # Test all methods
python -m ipol_runner test <method>     # Test specific method

# Check validation status
python -m ipol_runner status

# Generate web gallery (only shows validated methods)
python -m ipol_runner gallery -o ./output          # Generate gallery
python -m ipol_runner gallery -o ./output --open   # Generate and open in browser
```

## Validation Workflow

**Important**: Methods must pass validation before appearing in the gallery.

1. **Test a method**:
   ```bash
   python -m ipol_runner test <method>
   ```

2. **Check validation status**:
   ```bash
   python -m ipol_runner status
   ```
   Output shows passed/failed methods with error details for failures.

3. **Generate gallery**:
   ```bash
   python -m ipol_runner gallery -o ./output --open
   ```
   - Only validated (passed) methods appear in the main gallery
   - Failed methods are listed in a separate "Failed Validation" section with error details

4. **Validation status is tracked in** `validation_status.json`:
   - Passed methods: name and timestamp
   - Failed methods: name, error message, and optional notes

### Sample Inputs

Place sample inputs in `samples/`:
- `samples/<method_name>/` - Method-specific samples (highest priority)
- `samples/<input_type>/` - Generic samples by input type (e.g., `samples/image/`)
- `samples/test_image.png` - Auto-generated fallback for image methods

## Current Methods (2025)

| CLI Name | IPOL ID | Category | Input | Description |
|----------|---------|----------|-------|-------------|
| `qmsanet` | 545 | denoising | image | Quaternion denoising network |
| `kervrann` | 602 | change_detection | image_pair | Symmetric change detector |
| `cstrd` | 485 | detection | image | Tree ring detection |
| `gaussian_splatting` | 566 | 3d_reconstruction | image | Gaussian splatting |
| `latent_diffusion` | 580 | generation | image | Aerial image generation |
| `phase_unwrap` | 583 | phase_processing | image | Phase unwrapping |
| `nerf_specularity` | 562 | 3d_reconstruction | dataset_id | NeRF comparison |
| `semiogram` | 535 | medical | sensor_data | Gait analysis |
| `sign_lmsls` | 560a | segmentation | pose_data | Sign language (LMSLS) |
| `sign_asslisu` | 560b | segmentation | pose_data | Sign language (ASSLiSU) |

## Tips for Reproduction

1. **Read the IPOL article** - Understand inputs, outputs, and parameters
2. **Check the demo page** - IPOL demos show expected behavior
3. **Preserve original code** - Adapters wrap, don't modify original scripts
4. **Handle outputs properly** - Move/copy outputs to the specified output_dir
5. **Set reasonable timeouts** - Some methods take minutes to run
6. **Test with provided samples** - Most IPOL methods include test data
