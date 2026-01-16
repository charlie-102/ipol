"""VaxNeRF accelerated NeRF adapter (IPOL 2024 article 553)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class VaxNeRFMethod(IPOLMethod):
    """VaxNeRF: Visual hull accelerated Neural Radiance Fields."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_553_nerf_vaxnerf"

    # Pre-defined scenes with checkpoints
    SCENES = ["lego", "fern"]

    @property
    def name(self) -> str:
        return "nerf_vaxnerf"

    @property
    def display_name(self) -> str:
        return "VaxNeRF Accelerated NeRF"

    @property
    def description(self) -> str:
        return "Neural radiance fields with visual hull acceleration for faster rendering"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID

    @property
    def requires_cuda(self) -> bool:
        return True

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "scene": {
                "type": "choice",
                "choices": self.SCENES,
                "default": "lego",
                "description": "Pre-trained scene to render"
            },
            "elevation": {
                "type": "float",
                "default": 30.0,
                "min": -90.0,
                "max": 90.0,
                "description": "Camera elevation angle in degrees"
            },
            "azimuth": {
                "type": "float",
                "default": 20.0,
                "min": 0.0,
                "max": 360.0,
                "description": "Camera azimuth angle in degrees"
            },
            "radius": {
                "type": "float",
                "default": 1.0,
                "min": 0.5,
                "max": 1.5,
                "description": "Distance to scene center"
            },
            "scaling": {
                "type": "float",
                "default": 1.0,
                "min": 0.25,
                "max": 1.0,
                "description": "Image scaling factor"
            },
            "step": {
                "type": "int",
                "default": 2000,
                "min": 100,
                "max": 10000,
                "description": "Training step for checkpoint"
            },
            "use_vax": {
                "type": "bool",
                "default": True,
                "description": "Use VaxNeRF (visual hull acceleration)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run VaxNeRF rendering."""
        # Get parameters
        scene = params.get("scene", "lego")
        elevation = params.get("elevation", 30.0)
        azimuth = params.get("azimuth", 20.0)
        radius = params.get("radius", 1.0)
        scaling = params.get("scaling", 1.0)
        step = params.get("step", 2000)
        use_vax = params.get("use_vax", True)

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "query_demo.py"),
            "--scene", scene,
            "--elevation", str(elevation),
            "--azimuth", str(azimuth),
            "--radius", str(radius),
            "--scaling", str(scaling),
            "--step", str(step),
            "--out_dir", str(output_dir)
        ]
        if use_vax:
            cmd.append("--vax")

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
                    error_message=f"VaxNeRF failed: {result.stderr}"
                )

            # Check for outputs
            outputs = {}
            primary = None

            for out_file in output_dir.glob("*.png"):
                outputs[out_file.stem] = out_file
                if primary is None:
                    primary = out_file

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
                error_message="VaxNeRF timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
