"""MPRNet Image Restoration adapter (IPOL 2023 article 446)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class MPRNetMethod(IPOLMethod):
    """Multi-stage progressive image restoration network."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_446_mprnet"

    @property
    def name(self) -> str:
        return "mprnet"

    @property
    def display_name(self) -> str:
        return "MPRNet Image Restoration"

    @property
    def description(self) -> str:
        return "Deep multi-stage network for denoising, deblurring, and deraining"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    @property
    def requires_cuda(self) -> bool:
        return False  # Can run on CPU, MPS, or CUDA

    @property
    def supports_mps(self) -> bool:
        return True  # Supports Apple MPS backend

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "task": {
                "type": "choice",
                "choices": ["denoising", "deblurring", "deraining"],
                "default": "denoising",
                "description": "Restoration task type"
            },
            "add_noise": {
                "type": "bool",
                "default": False,
                "description": "Add synthetic noise before denoising"
            },
            "sigma": {
                "type": "int",
                "default": 25,
                "min": 1,
                "max": 100,
                "description": "Noise sigma (if add_noise=True)"
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
        """Run MPRNet restoration."""
        input_path = inputs[0]
        task = params.get("task", "denoising")
        add_noise = params.get("add_noise", False)
        sigma = params.get("sigma", 25)
        device = params.get("device", "cpu")

        # Select script based on task
        script_map = {
            "denoising": "denoising.py",
            "deblurring": "deblurring.py",
            "deraining": "deraining.py"
        }
        script = self.METHOD_DIR / script_map[task]

        # Check for model weights (try both naming conventions)
        weights_path = self.METHOD_DIR / "models" / f"model_{task}.pth"
        if not weights_path.exists():
            # Try alternate naming
            weights_path = self.METHOD_DIR / "models" / f"{task.capitalize()}.pth"
        if not weights_path.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Model weights not found. Download from IPOL and place in {self.METHOD_DIR}/models/"
            )

        # Build command - script uses sys.argv
        # For denoising: input, output, add_noise, [sigma], device
        # For deblurring/deraining: input, output, device
        if task == "denoising":
            cmd = [
                sys.executable,
                str(script),
                str(input_path),
                str(output_dir),
                "1" if add_noise else "0",
            ]
            if add_noise:
                cmd.append(str(sigma))
            cmd.append(device)
        else:
            cmd = [
                sys.executable,
                str(script),
                str(input_path),
                str(output_dir),
                device,
            ]

        try:
            # Longer timeout for CPU, shorter for GPU
            timeout = 120 if device in ["cuda", "mps"] else 600
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Check for outputs
            outputs = {}
            primary = None

            result_path = output_dir / "result.png"
            if result_path.exists():
                primary = result_path
                outputs["result"] = result_path

            noisy_path = output_dir / "noisy.png"
            if noisy_path.exists():
                outputs["noisy"] = noisy_path

            diff_path = output_dir / "diff.png"
            if diff_path.exists():
                outputs["diff"] = diff_path

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
                error_message=f"MPRNet failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Method timed out after {timeout}s (try using --param device=mps or device=cuda for faster processing)"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
