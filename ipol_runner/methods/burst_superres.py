"""Handheld Burst Super-Resolution adapter (IPOL 2023 article 460)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class BurstSuperResMethod(IPOLMethod):
    """Handheld burst super-resolution from multiple frames."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_460_burst_superres"

    @property
    def name(self) -> str:
        return "burst_superres"

    @property
    def display_name(self) -> str:
        return "Handheld Burst Super-Resolution"

    @property
    def description(self) -> str:
        return "Enhance resolution from multiple handheld camera frames (RAW DNG)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.VIDEO  # Expects directory of DNG files

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "scale": {
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 4.0,
                "description": "Upscaling factor"
            },
            "robustness": {
                "type": "bool",
                "default": False,
                "description": "Enable robustness mode for moving scenes"
            },
            "post_process": {
                "type": "bool",
                "default": False,
                "description": "Apply post-processing (sharpening, tonemapping)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run burst super-resolution."""
        input_path = inputs[0]  # Directory containing DNG files

        # Check for DNG files
        dng_files = list(input_path.glob("*.dng")) + list(input_path.glob("*.DNG"))
        if not dng_files:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="No DNG RAW files found. This method requires RAW camera files."
            )

        output_path = output_dir / "super_resolved.png"

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "demo.py"),
            "--impath", str(input_path),
            "--outpath", str(output_path),
            "--scale", str(params.get("scale", 2.0)),
            "--R_on", str(params.get("robustness", False)),
            "--post_process", str(params.get("post_process", False)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600  # Longer timeout for burst processing
            )

            if output_path.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_path,
                    outputs={"super_resolved": output_path}
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Burst super-resolution failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
