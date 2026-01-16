"""Fixed Pattern Noise Reduction adapter (IPOL preprint 436)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class FPNReductionMethod(IPOLMethod):
    """Fixed pattern noise reduction for infrared video sequences."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_436_fpn_reduction"

    @property
    def name(self) -> str:
        return "fpn_reduction"

    @property
    def display_name(self) -> str:
        return "Fixed Pattern Noise Reduction"

    @property
    def description(self) -> str:
        return "Recursive estimation of FPN offset for infrared image sequences"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

    @property
    def input_type(self) -> InputType:
        return InputType.VIDEO

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "noise_level": {
                "type": "float",
                "default": 10.0,
                "min": 0.0,
                "max": 100.0,
                "description": "Noise level (sigma) for offset noise"
            },
            "noise_level_signal": {
                "type": "float",
                "default": 25.0,
                "min": 0.0,
                "max": 100.0,
                "description": "Noise level for signal-dependent noise"
            },
            "add_noise": {
                "type": "bool",
                "default": False,
                "description": "Add synthetic noise to input"
            },
            "kernel": {
                "type": "choice",
                "choices": ["none", "average", "bilateral"],
                "default": "average",
                "description": "Filtering kernel type"
            },
            "size": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 15,
                "description": "Kernel size"
            },
            "threshold": {
                "type": "int",
                "default": 200,
                "min": 0,
                "max": 1000,
                "description": "Threshold for high-frequency components"
            },
            "m": {
                "type": "float",
                "default": 10.0,
                "min": 1.0,
                "max": 100.0,
                "description": "Averaging parameter M for offset estimation"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run FPN reduction using standalone Python script."""
        input_path = inputs[0]

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "fpn_reduction.py"),
            "--input", str(input_path),
            "--output", str(output_dir),
            "--noise_level", str(params.get("noise_level", 10.0)),
            "--noise_level_signal", str(params.get("noise_level_signal", 25.0)),
            "--kernel", params.get("kernel", "average"),
            "--size", str(params.get("size", 3)),
            "--threshold", str(params.get("threshold", 200)),
            "--M", str(params.get("m", 10.0)),
        ]

        if params.get("add_noise", False):
            cmd.append("--add_noise")

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

            # Collect output images
            for f in sorted(output_dir.glob("output_*.png")):
                if primary is None:
                    primary = f
                outputs[f.stem] = f

            # Check for metrics file
            metrics_file = output_dir / "metrics.txt"
            if metrics_file.exists():
                outputs["metrics"] = metrics_file

            if not primary:
                error_msg = result.stderr if result.stderr else result.stdout
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"FPN reduction failed: {error_msg[:500]}"
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
                error_message="FPN reduction timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
