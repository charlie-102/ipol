"""Video Signal-Dependent Noise Estimation adapter (IPOL 2023 article 420).

Estimates noise curves from consecutive video frames.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class VideoNoiseMethod(IPOLMethod):
    """Video signal-dependent noise estimation via inter-frame prediction."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_420_video_noise"

    @property
    def name(self) -> str:
        return "video_noise"

    @property
    def display_name(self) -> str:
        return "Video Noise Estimation"

    @property
    def description(self) -> str:
        return "Estimate signal-dependent noise curves from consecutive video frames"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION  # Noise estimation/detection

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR  # Two consecutive frames

    @property
    def input_count(self) -> int:
        return 2

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "bins": {
                "type": "int",
                "default": 16,
                "min": 4,
                "max": 64,
                "description": "Number of intensity bins"
            },
            "q": {
                "type": "float",
                "default": 0.01,
                "min": 0.001,
                "max": 0.1,
                "description": "Quantile of block pairs"
            },
            "s": {
                "type": "int",
                "default": 5,
                "min": 1,
                "max": 20,
                "description": "Search radius of patch matching"
            },
            "w": {
                "type": "int",
                "default": 8,
                "min": 4,
                "max": 32,
                "description": "Block size"
            },
            "grayscale": {
                "type": "bool",
                "default": False,
                "description": "Process images as grayscale"
            },
            "add_noise": {
                "type": "bool",
                "default": False,
                "description": "Add synthetic noise for testing"
            },
            "noise_a": {
                "type": "float",
                "default": 0.2,
                "min": 0.0,
                "max": 1.0,
                "description": "Noise model parameter a (if add_noise)"
            },
            "noise_b": {
                "type": "float",
                "default": 0.2,
                "min": 0.0,
                "max": 1.0,
                "description": "Noise model parameter b (if add_noise)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run video noise estimation."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Two consecutive frames required"
            )

        frame0 = inputs[0]
        frame1 = inputs[1]

        # Check input format - method requires TIFF/DNG
        valid_ext = {'.tif', '.tiff', '.dng'}
        if frame0.suffix.lower() not in valid_ext:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Input must be TIFF or DNG format, got {frame0.suffix}"
            )

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            str(frame0),
            str(frame1),
            "-bins", str(params.get("bins", 16)),
            "-q", str(params.get("q", 0.01)),
            "-s", str(params.get("s", 5)),
            "-w", str(params.get("w", 8)),
        ]

        if params.get("grayscale", False):
            cmd.append("-g")

        if params.get("add_noise", False):
            cmd.append("-add_noise")
            cmd.extend(["-noise_a", str(params.get("noise_a", 0.2))])
            cmd.extend(["-noise_b", str(params.get("noise_b", 0.2))])

        try:
            import os
            env = os.environ.copy()
            env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"

            result = subprocess.run(
                cmd,
                cwd=str(output_dir),  # Run in output dir so files are saved there
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            # Check for outputs
            outputs = {}
            primary = None

            curve_png = output_dir / "curve_s0.png"
            curve_txt = output_dir / "curve_s0.txt"

            if curve_png.exists():
                primary = curve_png
                outputs["noise_curve"] = curve_png

            if curve_txt.exists():
                outputs["noise_data"] = curve_txt

            if primary:
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=primary,
                    outputs=outputs
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Video noise estimation failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
