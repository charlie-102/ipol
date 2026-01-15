"""Testing utilities for IPOL methods."""
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import IPOLMethod, InputType
from .registry import get_all_methods
from .runner import run_method
from .validation import mark_passed, mark_failed, mark_skipped_cuda, mark_skipped_docker, print_validation_report


# Sample inputs directory (relative to ipol_runner)
SAMPLES_DIR = Path(__file__).parent.parent / "samples"


def get_sample_inputs(method: IPOLMethod) -> Optional[List[Path]]:
    """Get sample input files for a method.

    Looks for samples in:
    1. samples/<method_name>/  - method-specific samples
    2. samples/<input_type>/   - generic samples by input type
    """
    # Method-specific samples
    method_samples = SAMPLES_DIR / method.name
    if method_samples.exists():
        # Check for files first
        files = sorted(f for f in method_samples.iterdir() if f.is_file())
        if files:
            count = method.input_count if method.input_count > 0 else 1
            if len(files) >= count:
                return files[:count]
            return files

        # For POSE_DATA type, check for subdirectories (folder of skeleton files)
        if method.input_type == InputType.POSE_DATA:
            subdirs = sorted(d for d in method_samples.iterdir() if d.is_dir())
            if subdirs:
                # Return the first subdirectory as the input folder
                return [subdirs[0]]

    # Generic samples by input type
    type_samples = SAMPLES_DIR / method.input_type.value
    if type_samples.exists():
        files = sorted(f for f in type_samples.iterdir() if f.is_file())
        count = method.input_count if method.input_count > 0 else 1
        if len(files) >= count:
            return files[:count]

    # Built-in test images
    if method.input_type == InputType.IMAGE:
        test_img = SAMPLES_DIR / "test_image.png"
        if test_img.exists():
            return [test_img]

    return None


def create_test_image(path: Path, width: int = 256, height: int = 256) -> Path:
    """Create a simple test image (gradient pattern)."""
    try:
        import numpy as np
        try:
            import cv2
            # Create gradient test image
            img = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    img[y, x] = [x % 256, y % 256, (x + y) % 256]
            cv2.imwrite(str(path), img)
        except ImportError:
            from PIL import Image
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for y in range(height):
                for x in range(width):
                    pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
            img.save(path)
        return path
    except ImportError:
        return None


def test_method(
    method: IPOLMethod,
    verbose: bool = False,
    update_status: bool = True
) -> Tuple[bool, str]:
    """Test a single method with sample input.

    Args:
        method: The method to test
        verbose: Show verbose output
        update_status: Update validation_status.json

    Returns:
        Tuple of (success, message)
    """
    method_name = method.name
    print(f"Testing {method_name}...")

    # Check if method requires CUDA
    if method.requires_cuda:
        msg = f"[{method_name}] SKIP - Requires NVIDIA CUDA GPU"
        if update_status:
            mark_skipped_cuda(method_name)
        return False, msg

    # Check if method requires Docker
    if method.requires_docker:
        msg = f"[{method_name}] SKIP - Requires IPOL Docker infrastructure"
        if update_status:
            mark_skipped_docker(method_name)
        return False, msg

    # Handle DATASET_ID type - no file inputs needed
    if method.input_type == InputType.DATASET_ID:
        inputs = []
    else:
        # Get sample inputs
        inputs = get_sample_inputs(method)

        if not inputs:
            # For image methods, try to create a test image
            if method.input_type in (InputType.IMAGE, InputType.IMAGE_PAIR):
                SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                test_img = SAMPLES_DIR / "test_image.png"
                if not test_img.exists():
                    if not create_test_image(test_img):
                        msg = f"[{method_name}] SKIP - No sample inputs and cannot create test image"
                        if update_status:
                            mark_failed(method_name, "No sample inputs", "Need to add sample input files")
                        return False, msg

                if method.input_type == InputType.IMAGE:
                    inputs = [test_img]
                else:
                    inputs = [test_img, test_img]
            else:
                msg = f"[{method_name}] SKIP - No sample inputs for {method.input_type.value}"
                if update_status:
                    mark_failed(method_name, f"No sample inputs for {method.input_type.value}",
                               f"Need sample files in samples/{method_name}/ or samples/{method.input_type.value}/")
                return False, msg

    # Validate inputs exist
    for inp in inputs:
        if not inp.exists():
            msg = f"[{method_name}] SKIP - Sample input not found: {inp}"
            if update_status:
                mark_failed(method_name, f"Sample input not found: {inp}")
            return False, msg

    # Get default parameters
    params = {}
    for name, spec in method.get_parameters().items():
        if "default" in spec:
            params[name] = spec["default"]
        elif spec.get("required"):
            # For required params without default, we can't test
            msg = f"[{method_name}] SKIP - Required param '{name}' has no default"
            if update_status:
                mark_failed(method_name, f"Required param '{name}' has no default",
                           "Add default value or provide test params")
            return False, msg

    # Run in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()

        try:
            result = run_method(method, inputs, output_dir, params, verbose=verbose)

            if result.success:
                # Check outputs exist
                if result.primary_output and result.primary_output.exists():
                    msg = f"[{method_name}] PASS - Output: {result.primary_output.name}"
                    if update_status:
                        mark_passed(method_name, str(result.primary_output.name))
                    return True, msg
                elif result.outputs:
                    msg = f"[{method_name}] PASS - {len(result.outputs)} outputs"
                    if update_status:
                        mark_passed(method_name, f"{len(result.outputs)} outputs")
                    return True, msg
                else:
                    msg = f"[{method_name}] PASS - (no file output)"
                    if update_status:
                        mark_passed(method_name, "no file output")
                    return True, msg
            else:
                msg = f"[{method_name}] FAIL - {result.error_message}"
                if update_status:
                    mark_failed(method_name, result.error_message or "Unknown error")
                return False, msg

        except Exception as e:
            msg = f"[{method_name}] ERROR - {str(e)}"
            if update_status:
                mark_failed(method_name, str(e))
            return False, msg


def test_all_methods(verbose: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Test all registered methods.

    Returns:
        Dict mapping method name to (success, message) tuple
    """
    results = {}
    methods = get_all_methods()

    print(f"Testing {len(methods)} methods...\n")

    for name, method in methods.items():
        success, message = test_method(method, verbose=verbose)
        results[name] = (success, message)
        print(message)

    # Print validation report at end
    print_validation_report()

    return results


def validate_method_setup(method: IPOLMethod) -> List[str]:
    """Validate method is properly set up.

    Returns list of issues found.
    """
    issues = []

    # Check METHOD_DIR exists
    if hasattr(method, 'METHOD_DIR'):
        if not method.METHOD_DIR.exists():
            issues.append(f"METHOD_DIR not found: {method.METHOD_DIR}")

    # Check requirements file
    if method.requirements_file:
        if not method.requirements_file.exists():
            issues.append(f"Requirements file not found: {method.requirements_file}")

    # Check parameters have required fields
    for name, spec in method.get_parameters().items():
        if "type" not in spec:
            issues.append(f"Parameter '{name}' missing 'type' field")
        if spec.get("type") == "choice" and "choices" not in spec:
            issues.append(f"Choice parameter '{name}' missing 'choices' field")

    return issues
