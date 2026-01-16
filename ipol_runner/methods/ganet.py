"""GANet Stereo Matching adapter (IPOL 2023 article 441)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class GANetMethod(IPOLMethod):
    """Guided aggregation network for end-to-end stereo matching."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_441_ganet"

    @property
    def name(self) -> str:
        return "ganet"

    @property
    def display_name(self) -> str:
        return "GANet Stereo Matching"

    @property
    def description(self) -> str:
        return "Deep stereo matching using guided aggregation for disparity estimation"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR  # Left and right stereo images

    @property
    def input_count(self) -> int:
        return 2

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "max_disp": {
                "type": "int",
                "default": 192,
                "min": 48,
                "max": 384,
                "description": "Maximum disparity (multiple of 48)"
            },
            "crop_height": {
                "type": "int",
                "default": 192,
                "min": 48,
                "max": 384,
                "description": "Crop height for CPU mode (multiple of 48)"
            },
            "crop_width": {
                "type": "int",
                "default": 192,
                "min": 48,
                "max": 384,
                "description": "Crop width for CPU mode (multiple of 48)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run GANet stereo matching."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="GANet requires two input images (left and right stereo pair)"
            )

        left_path = inputs[0]
        right_path = inputs[1]
        output_path = output_dir / "disparity.png"

        # Check for model weights
        model_path = self.METHOD_DIR / "models" / "sceneflow_epoch_10.pth"
        if not model_path.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Model weights not found: {model_path}. Download from IPOL."
            )

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            str(left_path),
            str(right_path),
            str(output_path),
            "--max_disp", str(params.get("max_disp", 192)),
            "--crop_height", str(params.get("crop_height", 192)),
            "--crop_width", str(params.get("crop_width", 192)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )

            outputs = {}
            primary = None

            # Check for various output formats
            for pattern in ["disparity.png", "output.png", "output_color.png"]:
                out_file = output_dir / pattern
                if out_file.exists():
                    if primary is None:
                        primary = out_file
                    outputs[out_file.stem] = out_file

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
                error_message=f"GANet failed: {error_msg[:500]}"
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
