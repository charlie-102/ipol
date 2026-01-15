"""Gaussian Splatting adapter (IPOL 566)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class GaussianSplattingMethod(IPOLMethod):
    """3D Gaussian Splatting - fit Gaussians to reconstruct an image."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_566_gaussian_splatting"

    @property
    def name(self) -> str:
        return "gaussian_splatting"

    @property
    def display_name(self) -> str:
        return "Gaussian Splatting"

    @property
    def description(self) -> str:
        return "3D Gaussian Splatting to fit random Gaussians to a target image"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    @property
    def requires_cuda(self) -> bool:
        return True  # Requires NVIDIA GPU for gsplat CUDA kernels

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "num_points": {
                "type": "int",
                "default": 100000,
                "description": "Number of Gaussian splats"
            },
            "iterations": {
                "type": "int",
                "default": 1000,
                "description": "Training iterations"
            },
            "learning_rate": {
                "type": "float",
                "default": 0.01,
                "description": "Optimizer learning rate"
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
            sys.executable, str(self.METHOD_DIR / "main.py"),
            "--input", str(input_path),
            "--num_points", str(params.get("num_points", 100000)),
            "--iterations", str(params.get("iterations", 1000)),
            "--learning_rate", str(params.get("learning_rate", 0.01)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=600
            )

            outputs = {}
            primary = None

            # Check for outputs
            training_gif = output_dir / "training.gif"
            if training_gif.exists():
                outputs["training"] = training_gif
                primary = training_gif

            loss_curve = output_dir / "loss_curve.png"
            if loss_curve.exists():
                outputs["loss_curve"] = loss_curve

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
                error_message="Gaussian Splatting timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
