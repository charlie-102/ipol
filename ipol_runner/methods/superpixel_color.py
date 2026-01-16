"""Superpixel Color Transfer adapter (IPOL 2024 article 522).

Non-local matching of superpixel-based deep features for color transfer and colorization.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SuperpixelColorMethod(IPOLMethod):
    """Color transfer and colorization using superpixel-based deep features.

    Uses VGG-19 features for non-local matching between superpixels
    to transfer colors from a reference image to a target grayscale image.
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_522_superpixel_color"

    @property
    def name(self) -> str:
        return "superpixel_color"

    @property
    def display_name(self) -> str:
        return "Superpixel Color Transfer"

    @property
    def description(self) -> str:
        return "Non-local superpixel matching for color transfer and colorization"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.GENERATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR

    @property
    def input_count(self) -> int:
        return 2  # target (grayscale) and reference (color)

    @property
    def requires_cuda(self) -> bool:
        return True  # Uses VGG features on GPU

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "mode": {
                "type": "choice",
                "choices": ["colorization", "transfer"],
                "default": "colorization",
                "description": "Colorization: grayscale to color. Transfer: color to color."
            },
            "delta_s": {
                "type": "float",
                "default": 10.0,
                "min": 0.1,
                "max": 100.0,
                "description": "Spatial weight for superpixel matching"
            },
            "delta_c": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "description": "Color weight for superpixel matching"
            },
            "temperature": {
                "type": "float",
                "default": 0.015,
                "min": 0.001,
                "max": 1.0,
                "description": "Softmax temperature for attention"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run superpixel-based color transfer."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Requires 2 inputs: target image and reference image"
            )

        target_path = inputs[0]
        reference_path = inputs[1]

        # Get parameters
        mode = params.get("mode", "colorization")
        delta_s = params.get("delta_s", 10.0)
        delta_c = params.get("delta_c", 0.1)
        temperature = params.get("temperature", 0.015)

        # Choose script based on mode
        if mode == "colorization":
            script = self.METHOD_DIR / "colorization.py"
            output_file = output_dir / "colorized.png"
        else:
            script = self.METHOD_DIR / "transfer.py"
            output_file = output_dir / "transferred.png"

        model_path = self.METHOD_DIR / "models" / "vgg19_bn.pth"

        # Build command
        cmd = [
            sys.executable,
            str(script),
            "--target", str(target_path),
            "--reference", str(reference_path),
            "--deltaS", str(delta_s),
            "--deltaC", str(delta_c),
            "--temp", str(temperature),
            "--output", str(output_file),
            "--model", str(model_path)
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
                    error_message=f"Color transfer failed: {result.stderr}"
                )

            outputs = {}
            primary = None

            if output_file.exists():
                outputs["result"] = output_file
                primary = output_file

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
                error_message="Color transfer timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
