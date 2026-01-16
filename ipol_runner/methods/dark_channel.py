"""Dark Channel Prior dehazing adapter (IPOL 2024 article 530).

Supports both C++ (original) and Python backends.
"""
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class DarkChannelMethod(IPOLMethod):
    """Image dehazing using dark channel prior.

    Supports two backends:
    - cpp: Original C++ implementation (faster, requires compilation)
    - python: Pure Python implementation (no compilation needed)
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_530_dark_channel"
    SRC_DIR = METHOD_DIR / "src"

    @property
    def name(self) -> str:
        return "dark_channel"

    @property
    def display_name(self) -> str:
        return "Dark Channel Prior Dehazing"

    @property
    def description(self) -> str:
        return "Single image dehazing using dark channel prior (C++ or Python backend)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

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
                "description": "Backend: python (pure Python) or cpp (original C++, requires compilation)"
            },
            "omega": {
                "type": "float",
                "default": 0.95,
                "min": 0.0,
                "max": 1.0,
                "description": "Haze removal amount (0=none, 1=full)"
            },
            "patch_radius": {
                "type": "int",
                "default": 7,
                "min": 1,
                "max": 50,
                "description": "Patch radius for dark channel computation"
            },
            "guided_radius": {
                "type": "int",
                "default": 30,
                "min": 1,
                "max": 100,
                "description": "Radius for guided filter refinement"
            },
            "t0": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "description": "Minimum transmission value"
            }
        }

    def _ensure_compiled(self) -> tuple[bool, str]:
        """Ensure the C++ binary is compiled. Returns (success, error_msg)."""
        binary = self.SRC_DIR / "dehazeDCP"

        if binary.exists():
            return True, ""

        # Try to compile
        try:
            result = subprocess.run(
                ["make"],
                cwd=str(self.SRC_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return False, f"Compilation failed: {result.stderr}"

            if not binary.exists():
                return False, "Compilation succeeded but binary not found"

            return True, ""

        except FileNotFoundError:
            return False, "make not found. Install build tools or use python backend."
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
        # Ensure binary is compiled
        compiled, error = self._ensure_compiled()
        if not compiled:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"C++ backend not available: {error}. Try backend=python."
            )

        # Get parameters
        omega = params.get("omega", 0.95)
        patch_radius = params.get("patch_radius", 7)
        guided_radius = params.get("guided_radius", 30)
        t0 = params.get("t0", 0.1)

        # Output files
        output_dehazed = output_dir / "dehazed.png"
        output_transmission = output_dir / "transmission.png"

        # Build command
        binary = self.SRC_DIR / "dehazeDCP"
        cmd = [
            str(binary),
            str(input_path),
            str(output_dehazed),
            "-w", str(omega),
            "-s", str(patch_radius),
            "-r", str(guided_radius),
            "-t", str(t0),
            "-f", str(output_transmission)  # Output refined transmission
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.SRC_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Dehazing failed: {result.stderr}"
                )

            # Check for outputs
            outputs = {}
            primary = None

            if output_dehazed.exists():
                outputs["dehazed"] = output_dehazed
                primary = output_dehazed

            if output_transmission.exists():
                outputs["transmission"] = output_transmission

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
                error_message="Dehazing timed out after 300s"
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
        """Run using pure Python backend."""
        try:
            import numpy as np
            import imageio.v2 as imageio
            from .dark_channel_python import dehaze_dark_channel_prior

            # Load input image
            image = imageio.imread(str(input_path))

            # Get parameters
            omega = params.get("omega", 0.95)
            patch_radius = params.get("patch_radius", 7)
            guided_radius = params.get("guided_radius", 30)
            t0 = params.get("t0", 0.1)

            # Run dehazing
            dehazed, transmission, ambient = dehaze_dark_channel_prior(
                image,
                omega=omega,
                patch_radius=patch_radius,
                guided_radius=guided_radius,
                t0=t0
            )

            # Save outputs
            output_dehazed = output_dir / "dehazed.png"
            output_transmission = output_dir / "transmission.png"

            imageio.imwrite(str(output_dehazed), dehazed.astype(np.uint8))
            imageio.imwrite(
                str(output_transmission),
                (transmission * 255).astype(np.uint8)
            )

            outputs = {
                "dehazed": output_dehazed,
                "transmission": output_transmission
            }

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=output_dehazed,
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
                error_message=f"Python dehazing failed: {str(e)}\n{traceback.format_exc()}"
            )

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run dark channel prior dehazing."""
        input_path = inputs[0]
        backend = params.get("backend", "python")

        if backend == "cpp":
            return self._run_cpp(input_path, output_dir, params)
        else:
            return self._run_python(input_path, output_dir, params)
