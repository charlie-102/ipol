"""STORM microscopy super-resolution adapter (IPOL 2024 article 496)."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class STORMMethod(IPOLMethod):
    """STORM super-resolution localization microscopy."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_496_storm"

    @property
    def name(self) -> str:
        return "storm"

    @property
    def display_name(self) -> str:
        return "STORM Super-Resolution Microscopy"

    @property
    def description(self) -> str:
        return "Single-molecule localization microscopy for super-resolution imaging"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE  # Image stack (TIFF with multiple frames)

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "init_params": {
                "type": "str",
                "default": "100,2,2,1,1",
                "description": "Initial PSF parameters (amplitude,sigma_x,sigma_y,offset_x,offset_y)"
            },
            "threshold": {
                "type": "int",
                "default": 50,
                "min": 1,
                "max": 255,
                "description": "Detection threshold"
            },
            "psf_radius": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "PSF radius for detection"
            },
            "window_size": {
                "type": "int",
                "default": 7,
                "min": 3,
                "max": 21,
                "description": "Window size for sub-pixel localization"
            },
            "neighbourhood_size": {
                "type": "int",
                "default": 15,
                "min": 3,
                "max": 31,
                "description": "Neighbourhood size for center-of-mass detection"
            },
            "reconstruction_type": {
                "type": "choice",
                "choices": ["rec_crude", "rec_sub"],
                "default": "rec_sub",
                "description": "Reconstruction type (crude or sub-pixel)"
            },
            "crude_localization_type": {
                "type": "choice",
                "choices": ["crude_plm", "crude_com", "crude_bl"],
                "default": "crude_plm",
                "description": "Crude localization method (peak local maxima, center of mass, blob)"
            },
            "sub_pixel_localization_type": {
                "type": "choice",
                "choices": ["sub_1d", "sub_2d", "sub_mle"],
                "default": "sub_2d",
                "description": "Sub-pixel localization method (1D/2D least squares, MLE)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run STORM super-resolution localization."""
        input_path = inputs[0]

        # Get parameters
        init_params = params.get("init_params", "100,2,2,1,1")
        threshold = params.get("threshold", 50)
        psf_radius = params.get("psf_radius", 3)
        window_size = params.get("window_size", 7)
        neighbourhood_size = params.get("neighbourhood_size", 15)
        reconstruction_type = params.get("reconstruction_type", "rec_sub")
        crude_type = params.get("crude_localization_type", "crude_plm")
        sub_type = params.get("sub_pixel_localization_type", "sub_2d")

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            "--input", str(input_path),
            "--init_params", init_params,
            "--threshold", str(threshold),
            "--psf_radius", str(psf_radius),
            "--window_size", str(window_size),
            "--neighbourhood_size", str(neighbourhood_size),
            "--reconstruction_type", reconstruction_type,
            "--crude_localization_type", crude_type,
            "--sub_pixel_localization_type", sub_type
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
                    error_message=f"STORM failed: {result.stderr}"
                )

            # Move outputs from ./logs to output_dir
            logs_dir = self.METHOD_DIR / "logs"
            outputs = {}
            primary = None

            if logs_dir.exists():
                for out_file in logs_dir.glob("*"):
                    dst = output_dir / out_file.name
                    shutil.copy(out_file, dst)
                    outputs[out_file.stem] = dst
                    if out_file.suffix == ".png" and primary is None:
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
                error_message="STORM timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
