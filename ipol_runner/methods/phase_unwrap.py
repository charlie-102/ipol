"""Phase Unwrapping adapter (IPOL 583)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
import math

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class PhaseUnwrapMethod(IPOLMethod):
    """L1-Norm Redundant Delaunay Phase Unwrapping."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_583_phase_unwrap"

    @property
    def name(self) -> str:
        return "phase_unwrap"

    @property
    def display_name(self) -> str:
        return "Phase Unwrapping (L1-Norm Delaunay)"

    @property
    def description(self) -> str:
        return "Unwrap wrapped phase images using Delaunay triangulation"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.PHASE_PROCESSING

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "k": {
                "type": "int",
                "default": 0,
                "description": "Number of redundant edges for Delaunay graph"
            },
            "h": {
                "type": "float",
                "default": math.pi,
                "description": "Half-period of wrapping function"
            },
            "useMCF": {
                "type": "bool",
                "default": True,
                "description": "Use minimum cost flow algorithm"
            },
            "useSmallBasis": {
                "type": "bool",
                "default": True,
                "description": "Use small cycle basis detection"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        input_path = inputs[0]

        cmd = [
            sys.executable, str(self.METHOD_DIR / "run.py"),
            str(input_path),
            str(input_path),  # phi_orig_path (same as input)
            f"--k={params.get('k', 0)}",
            f"--h={params.get('h', math.pi)}",
            f"--useMCF={'true' if params.get('useMCF', True) else 'false'}",
            f"--useSmallBasis={'true' if params.get('useSmallBasis', True) else 'false'}",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=300
            )

            outputs = {}
            primary = None

            # Check for outputs
            result_tif = output_dir / "result_raster.tif"
            if result_tif.exists():
                outputs["unwrapped"] = result_tif
                primary = result_tif

            output_png = output_dir / "output_img.png"
            if output_png.exists():
                outputs["visualization"] = output_png
                if not primary:
                    primary = output_png

            input_png = output_dir / "input_img.png"
            if input_png.exists():
                outputs["input_viz"] = input_png

            if not primary:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stderr: {result.stderr}"
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
                error_message="Phase Unwrap timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
