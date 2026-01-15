"""Latent Diffusion for Aerial Imagery adapter (IPOL 580)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class LatentDiffusionMethod(IPOLMethod):
    """Latent Diffusion Model for generating aerial imagery from maps."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_580_latent_diffusion"

    @property
    def name(self) -> str:
        return "latent_diffusion"

    @property
    def display_name(self) -> str:
        return "Latent Diffusion Aerial Imagery"

    @property
    def description(self) -> str:
        return "Generate synthetic aerial images from map images using diffusion"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.GENERATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    @property
    def requires_cuda(self) -> bool:
        return True  # Requires NVIDIA GPU for diffusion model

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "time_steps": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 2000,
                "description": "Denoising steps (higher = better quality, slower)"
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
            sys.executable, str(self.METHOD_DIR / "run_demo.py"),
            "--img_path", str(input_path),
            "--time_steps", str(params.get("time_steps", 1000)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=900
            )

            # Move outputs from demo_output to output_dir
            demo_output = self.METHOD_DIR / "demo_output"
            outputs = {}
            primary = None

            if demo_output.exists():
                import shutil
                for f in demo_output.iterdir():
                    dst = output_dir / f.name
                    shutil.copy(f, dst)
                    outputs[f.stem] = dst
                    if f.name == "output_00.png":
                        primary = dst

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
                error_message="Latent Diffusion timed out after 900s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
