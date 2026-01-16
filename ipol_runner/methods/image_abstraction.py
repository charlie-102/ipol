"""Image Abstraction adapter (IPOL 2024 article 495).

Supports both C++ (original Tree of Shapes) and Python (simplified) backends.

Note: The Python backend is a simplified approximation. For full Tree of Shapes
functionality (watercolor, shaking, style transfer), use the C++ backend.
"""
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class ImageAbstractionMethod(IPOLMethod):
    """Structured abstraction of images using tree of shapes.

    Supports two backends:
    - cpp: Original C++ implementation with Qt4 (full features)
    - python: Simplified Python implementation (basic abstraction only)
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_495_image_abstraction"

    # Task types
    TASKS = {
        "abstraction": 0,
        "watercolor": 1,
        "shaking": 2,
        "smoothing": 3,
        "style_transfer": 4
    }

    # Synthesis models (for C++ backend)
    MODELS = {
        "original": 0,
        "ellipse": 1,
        "rectangle": 2,
        "circle": 3,
        "dictionary": 4
    }

    @property
    def name(self) -> str:
        return "image_abstraction"

    @property
    def display_name(self) -> str:
        return "Image Abstraction"

    @property
    def description(self) -> str:
        return "Structured abstraction using tree of shapes (C++ or simplified Python)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.GENERATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requires_compilation(self) -> bool:
        return False  # Python backend doesn't require compilation

    @property
    def requirements_file(self):
        return None  # Uses scipy, numpy from main requirements

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "backend": {
                "type": "choice",
                "choices": ["python", "cpp"],
                "default": "python",
                "description": "Backend: python (simplified) or cpp (full Tree of Shapes, requires Qt4)"
            },
            "task": {
                "type": "choice",
                "choices": list(self.TASKS.keys()),
                "default": "abstraction",
                "description": "Abstraction task (style_transfer requires C++ backend)"
            },
            "model": {
                "type": "choice",
                "choices": ["original", "ellipse", "rectangle", "circle"],
                "default": "ellipse",
                "description": "Shape synthesis model (C++ backend only)"
            },
            "threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Effect strength threshold"
            }
        }

    def _ensure_compiled(self) -> tuple[bool, str]:
        """Ensure the C++ binary is compiled. Returns (success, error_msg)."""
        binary = self.METHOD_DIR / "image_abstraction"

        if binary.exists():
            return True, ""

        # Try to compile with qmake
        try:
            # First run qmake
            result = subprocess.run(
                ["qmake", "-makefile", "."],
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return False, f"qmake failed: {result.stderr}. Install Qt4: brew install qt@4 or apt install qt4-qmake"

            # Then run make
            result = subprocess.run(
                ["make"],
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return False, f"make failed: {result.stderr}"

            if not binary.exists():
                return False, "Compilation succeeded but binary not found"

            return True, ""

        except FileNotFoundError as e:
            return False, f"Build tool not found: {e}. Install qmake and make, or use python backend."
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"

    def _run_cpp(
        self,
        input_path: Path,
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run using C++ backend."""
        import shutil

        # Ensure binary is compiled
        compiled, error = self._ensure_compiled()
        if not compiled:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"C++ backend not available: {error}. Try backend=python."
            )

        # Get parameters
        task = self.TASKS.get(params.get("task", "abstraction"), 0)
        model = self.MODELS.get(params.get("model", "ellipse"), 1)
        use_seg = "false"

        binary = self.METHOD_DIR / "image_abstraction"

        # Build command with default parameters
        cmd = [
            str(binary),
            str(input_path),
            str(task),      # Task
            str(model),     # Model
            use_seg,        # Segmentation
            "0",            # Alternative model
            # Default parameters for abstraction (6-14)
            "false", "1", "0.01", "100", "3", "0.5", "0", "1", "0",
            # Default parameters for watercolor (15-23)
            "false", "0", "0", "100", "3", "0.6", "0", "0", "0",
            # Default parameters for shaking (24-32)
            "false", "0", "0", "100", "3", "0.6", "0", "0", "0",
            # Default parameters for filtering (33-42)
            "false", "0", "0.1", "100", "3", "0.8", "0", "0", "0",
            # Default parameters for style transfer (42-55)
            "false", "0", "0", "100", "3", "0", "0", "1", "0.2", "2", "1", "0", "0", "0",
            # Style image and mask (dummy paths)
            str(input_path),
            str(input_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Image abstraction failed: {result.stderr}"
                )

            # Find and copy output files
            outputs = {}
            primary = None

            # Check for result.png (the main output)
            result_file = self.METHOD_DIR / "result.png"
            if result_file.exists():
                dst = output_dir / "abstraction.png"
                shutil.copy(result_file, dst)
                outputs["abstraction"] = dst
                primary = dst
                result_file.unlink()

            # Copy any other output files
            for out_file in self.METHOD_DIR.glob("output*.png"):
                dst = output_dir / out_file.name
                shutil.copy(out_file, dst)
                outputs[out_file.stem] = dst
                if primary is None:
                    primary = dst
                out_file.unlink()

            if not primary:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stdout: {result.stdout}, stderr: {result.stderr}"
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
                error_message="Image abstraction timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )

    def _run_python(
        self,
        input_path: Path,
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run using pure Python backend (simplified abstraction)."""
        try:
            import numpy as np
            import imageio.v2 as imageio
            from .image_abstraction_python import image_abstraction

            # Load input image
            image = imageio.imread(str(input_path))

            # Get parameters
            task = params.get("task", "abstraction")
            threshold = params.get("threshold", 0.5)

            # Check for unsupported tasks
            if task == "style_transfer":
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message="Style transfer requires C++ backend with Qt4. Use backend=cpp."
                )

            # Run abstraction
            result = image_abstraction(
                image,
                task=task,
                threshold=threshold
            )

            # Save output
            output_file = output_dir / "abstraction.png"
            imageio.imwrite(str(output_file), result.astype(np.uint8))

            outputs = {"abstraction": output_file}

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=output_file,
                outputs=outputs
            )

        except ImportError as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Missing dependency: {e}. Install with: pip install numpy scipy imageio"
            )
        except Exception as e:
            import traceback
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Python abstraction failed: {str(e)}\n{traceback.format_exc()}"
            )

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run image abstraction."""
        input_path = inputs[0]
        backend = params.get("backend", "python")

        if backend == "cpp":
            return self._run_cpp(input_path, output_dir, params)
        else:
            return self._run_python(input_path, output_dir, params)
