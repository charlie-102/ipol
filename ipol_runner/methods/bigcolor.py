"""BigColor image colorization adapter (IPOL 2024 article 542)."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class BigColorMethod(IPOLMethod):
    """BigColor deep learning colorization with BigGAN prior."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_542_bigcolor"

    @property
    def name(self) -> str:
        return "bigcolor"

    @property
    def display_name(self) -> str:
        return "BigColor Image Colorization"

    @property
    def description(self) -> str:
        return "Automatic grayscale image colorization using BigGAN prior"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.GENERATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requires_cuda(self) -> bool:
        return True

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "seed": {
                "type": "int",
                "default": -1,
                "min": -1,
                "max": 10000,
                "description": "Random seed (-1 for random)"
            },
            "epoch": {
                "type": "int",
                "default": 11,
                "min": 0,
                "max": 100,
                "description": "Model checkpoint epoch"
            },
            "type_resize": {
                "type": "choice",
                "choices": ["absolute", "original", "square", "powerof"],
                "default": "powerof",
                "description": "Image resize strategy"
            },
            "size_target": {
                "type": "int",
                "default": 256,
                "min": 64,
                "max": 512,
                "description": "Target size for processing"
            },
            "topk": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Top-k classes for colorization"
            },
            "use_ema": {
                "type": "bool",
                "default": False,
                "description": "Use exponential moving average weights"
            },
            "device": {
                "type": "choice",
                "choices": ["cuda:0", "cpu"],
                "default": "cpu",
                "description": "Device for inference"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run BigColor colorization."""
        input_path = inputs[0]

        # Create input directory and copy image
        input_dir = output_dir / "input_grays"
        input_dir.mkdir(exist_ok=True)
        shutil.copy(input_path, input_dir / input_path.name)

        # Get parameters
        seed = params.get("seed", -1)
        epoch = params.get("epoch", 11)
        type_resize = params.get("type_resize", "powerof")
        size_target = params.get("size_target", 256)
        topk = params.get("topk", 1)
        use_ema = params.get("use_ema", False)
        device = params.get("device", "cpu")

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "colorize_real.py"),
            "--seed", str(seed),
            "--epoch", str(epoch),
            "--type_resize", type_resize,
            "--size_target", str(size_target),
            "--topk", str(topk),
            "--device", device,
            "--path_input", str(input_dir),
            "--path_output", str(output_dir)
        ]
        if use_ema:
            cmd.append("--use_ema")

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
                    error_message=f"BigColor failed: {result.stderr}"
                )

            # Check for outputs
            outputs = {}
            primary = None

            output_file = output_dir / "output.jpg"
            if output_file.exists():
                outputs["colorized"] = output_file
                primary = output_file

            gray_file = output_dir / "gray.png"
            if gray_file.exists():
                outputs["gray"] = gray_file

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
                error_message="BigColor timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
