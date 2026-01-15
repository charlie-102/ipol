"""CS-TRD tree ring detection adapter."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class CSTRDMethod(IPOLMethod):
    """CS-TRD tree ring detection for wood cross-sections."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_485_cstrd"

    @property
    def name(self) -> str:
        return "cstrd"

    @property
    def display_name(self) -> str:
        return "CS-TRD Tree Ring Detection"

    @property
    def description(self) -> str:
        return "Detect tree rings in wood cross-section images"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "cx": {
                "type": "int",
                "default": 128,  # Center of typical test image
                "description": "Pith center X coordinate"
            },
            "cy": {
                "type": "int",
                "default": 128,  # Center of typical test image
                "description": "Pith center Y coordinate"
            },
            "sigma": {
                "type": "float",
                "default": 3.0,
                "min": 1.0,
                "max": 10.0,
                "description": "Gaussian filter parameter"
            },
            "nr": {
                "type": "int",
                "default": 360,
                "description": "Number of rays for sampling"
            },
            "alpha": {
                "type": "int",
                "default": 30,
                "min": 0,
                "max": 90,
                "description": "Collinearity threshold"
            },
            "th_low": {
                "type": "float",
                "default": 5.0,
                "description": "Low gradient threshold"
            },
            "th_high": {
                "type": "float",
                "default": 20.0,
                "description": "High gradient threshold"
            },
            "hsize": {
                "type": "int",
                "default": 0,
                "description": "Resize height (0 = original)"
            },
            "wsize": {
                "type": "int",
                "default": 0,
                "description": "Resize width (0 = original)"
            },
            "save_imgs": {
                "type": "bool",
                "default": True,
                "description": "Save intermediate visualization images"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters (cx and cy have defaults now)."""
        return None

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run CS-TRD tree ring detection."""
        input_path = inputs[0]

        script_path = self.METHOD_DIR / "main.py"

        # Build command
        cmd = [
            sys.executable, str(script_path),
            "--input", str(input_path),
            "--cx", str(params.get("cx", 128)),
            "--cy", str(params.get("cy", 128)),
            "--root", str(self.METHOD_DIR),
            "--output_dir", str(output_dir),
            "--sigma", str(params.get("sigma", 3.0)),
            "--nr", str(params.get("nr", 360)),
            "--alpha", str(params.get("alpha", 30)),
            "--th_low", str(params.get("th_low", 5.0)),
            "--th_high", str(params.get("th_high", 20.0)),
        ]

        hsize = params.get("hsize", 0)
        wsize = params.get("wsize", 0)
        if hsize > 0:
            cmd.extend(["--hsize", str(hsize)])
        if wsize > 0:
            cmd.extend(["--wsize", str(wsize)])

        if params.get("save_imgs", True):
            cmd.extend(["--save_imgs", "1"])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check for failure file (CS-TRD writes errors there)
            failure_file = self.METHOD_DIR / "demo_failure.txt"
            if failure_file.exists():
                with open(failure_file) as f:
                    error_msg = f.read()
                failure_file.unlink()
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"CS-TRD failed: {error_msg}"
                )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"CS-TRD failed: {result.stderr}"
                )

            outputs = {}
            primary = None

            # Check for outputs
            output_png = output_dir / "output.png"
            if output_png.exists():
                outputs["rings"] = output_png
                primary = output_png

            labelme_json = output_dir / "labelme.json"
            if labelme_json.exists():
                outputs["rings_json"] = labelme_json

            # Intermediate images if saved
            for name in ["preprocessing", "edges", "filter", "chains", "connect", "postprocessing"]:
                img_path = output_dir / f"{name}.png"
                if img_path.exists():
                    outputs[name] = img_path

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
                error_message="CS-TRD timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
