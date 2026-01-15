"""Kervrann change detection adapter."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class KervrannMethod(IPOLMethod):
    """Kervrann symmetric change detection for image pairs."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_602_kervrann"

    @property
    def name(self) -> str:
        return "kervrann"

    @property
    def display_name(self) -> str:
        return "Kervrann Change Detector"

    @property
    def description(self) -> str:
        return "Symmetric change detection using hypothesis testing (a contrario)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.CHANGE_DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR

    @property
    def input_count(self) -> int:
        return 2

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "scale": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 10,
                "description": "Number of scales"
            },
            "b": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 15,
                "description": "Side of square neighborhood"
            },
            "B": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 15,
                "description": "Side of square search window"
            },
            "metric": {
                "type": "choice",
                "choices": ["corr", "rho", "mult", "zncc", "lin"],
                "default": "lin",
                "description": "Dissimilarity measure"
            },
            "epsilon": {
                "type": "float",
                "default": 1.0,
                "min": 0,
                "description": "False alarm threshold"
            },
            "sigma": {
                "type": "float",
                "default": 0.8,
                "min": 0,
                "description": "Blur kernel standard deviation"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run Kervrann change detection."""
        image1 = inputs[0]
        image2 = inputs[1]

        script_path = self.METHOD_DIR / "ipol_kervrann.py"

        cmd = [
            sys.executable, str(script_path),
            "--image1", str(image1),
            "--image2", str(image2),
            "--dirout", str(output_dir),
            "--scale", str(params.get("scale", 2)),
            "--b", str(params.get("b", 3)),
            "--B", str(params.get("B", 3)),
            "--metric", str(params.get("metric", "lin")),
            "--epsilon", str(params.get("epsilon", 1.0)),
            "--sigma", str(params.get("sigma", 0.8)),
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
                    error_message=f"Kervrann failed: {result.stderr}"
                )

            outputs = {}
            primary = None

            # Check for outputs
            huvl = output_dir / "huvl.png"
            if huvl.exists():
                outputs["change_map"] = huvl
                primary = huvl

            pfal = output_dir / "pfal.png"
            if pfal.exists():
                outputs["pfa_map"] = pfal

            im1_out = output_dir / "im1.png"
            if im1_out.exists():
                outputs["image1_normalized"] = im1_out

            im2_out = output_dir / "im2.png"
            if im2_out.exists():
                outputs["image2_normalized"] = im2_out

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
                error_message="Kervrann timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
