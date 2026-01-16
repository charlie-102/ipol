"""iColoriT interactive colorization adapter (IPOL 2024 article 539)."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class IColoriTMethod(IPOLMethod):
    """iColoriT interactive image colorization with hints."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_539_icolorit"

    @property
    def name(self) -> str:
        return "icolorit"

    @property
    def display_name(self) -> str:
        return "iColoriT Interactive Colorization"

    @property
    def description(self) -> str:
        return "Interactive grayscale image colorization with optional color hints"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.GENERATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requires_cuda(self) -> bool:
        return False  # Can run on CPU, MPS, or CUDA

    @property
    def supports_mps(self) -> bool:
        return True  # Supports Apple MPS backend

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "model": {
                "type": "choice",
                "choices": ["icolorit_base_4ch_patch16_224"],
                "default": "icolorit_base_4ch_patch16_224",
                "description": "Model architecture"
            },
            "input_size": {
                "type": "int",
                "default": 224,
                "min": 64,
                "max": 512,
                "description": "Input image size"
            },
            "hint_size": {
                "type": "int",
                "default": 2,
                "min": 1,
                "max": 10,
                "description": "Size of hint region"
            },
            "val_hint_list": {
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 100,
                "description": "Number of color hints (0 for automatic)"
            },
            "device": {
                "type": "choice",
                "choices": ["cuda", "mps", "cpu"],
                "default": "cpu",
                "description": "Device for inference (cuda, mps for Apple Silicon, or cpu)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run iColoriT colorization."""
        input_path = inputs[0]

        # Create input directory
        input_dir = output_dir / "imgs"
        input_dir.mkdir(exist_ok=True)
        shutil.copy(input_path, input_dir / input_path.name)

        # Get parameters
        model = params.get("model", "icolorit_base_4ch_patch16_224")
        input_size = params.get("input_size", 224)
        hint_size = params.get("hint_size", 2)
        val_hint_list = params.get("val_hint_list", 0)
        device = params.get("device", "cuda")

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "infer.py"),
            "--model", model,
            "--model_path", str(self.METHOD_DIR / f"{model}.pth"),
            "--val_data_path", str(input_dir),
            "--pred_dir", str(output_dir),
            "--input_size", str(input_size),
            "--hint_size", str(hint_size),
            "--val_hint_list", str(val_hint_list),
            "--device", device,
            "--batch_size", "1"
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
                    error_message=f"iColoriT failed: {result.stderr}"
                )

            # Check for outputs
            outputs = {}
            primary = None

            for out_file in output_dir.glob("*.png"):
                if out_file.name != input_path.name:
                    outputs[out_file.stem] = out_file
                    if primary is None:
                        primary = out_file

            for out_file in output_dir.glob("*.jpg"):
                outputs[out_file.stem] = out_file
                if primary is None:
                    primary = out_file

            # Cleanup input dir
            shutil.rmtree(input_dir, ignore_errors=True)

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
                error_message="iColoriT timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
