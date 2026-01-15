"""QMSANet color image denoising adapter."""
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class QMSANetMethod(IPOLMethod):
    """QMSANet quaternion-based color image denoising."""

    MODELS = ["S15", "S25", "S35", "S50", "B25"]
    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_545_qmsanet"

    @property
    def name(self) -> str:
        return "qmsanet"

    @property
    def display_name(self) -> str:
        return "QMSANet Color Image Denoising"

    @property
    def description(self) -> str:
        return "Deep learning denoising using quaternion multi-scale attention network"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "pyproject.toml"  # Uses pyproject.toml

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "model": {
                "type": "choice",
                "choices": self.MODELS,
                "default": "S25",
                "description": "Pre-trained model (S=specific noise level, B=blind)"
            },
            "add_noise": {
                "type": "bool",
                "default": False,
                "description": "Add synthetic noise to input image"
            },
            "noise_level": {
                "type": "float",
                "default": 25.0,
                "min": 0,
                "max": 100,
                "description": "Noise level (sigma) when add_noise is True"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run QMSANet denoising."""
        import subprocess

        input_path = inputs[0]
        model = params.get("model", "S25")
        add_noise = params.get("add_noise", False)
        noise_level = params.get("noise_level", 25.0)

        # Build command
        script_path = self.METHOD_DIR / "color_noisy" / "syn_demo.py"
        cmd = [
            sys.executable, str(script_path),
            "--single_image", str(input_path),
            "--logdir", model,
            "--test_noiseL", str(noise_level),
        ]
        if add_noise:
            cmd.append("--add_noise")

        # Run in the color_noisy directory so it finds the model weights
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR / "color_noisy"),
                capture_output=True,
                text=True,
                timeout=600  # CPU inference can be slow
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"QMSANet failed: {result.stderr}"
                )

            # Move output files to output_dir
            src_output = self.METHOD_DIR / "color_noisy" / "output.png"
            src_results = self.METHOD_DIR / "color_noisy" / "results.txt"

            outputs = {}
            primary = None

            if src_output.exists():
                import shutil
                dst = output_dir / "output.png"
                shutil.copy(src_output, dst)
                outputs["denoised"] = dst
                primary = dst
                src_output.unlink()

            metrics = {}
            if src_results.exists():
                with open(src_results) as f:
                    content = f.read()
                    # Parse "PSNR: X, SSIM: Y"
                    for part in content.split(","):
                        if ":" in part:
                            k, v = part.split(":")
                            try:
                                metrics[k.strip()] = float(v.strip())
                            except ValueError:
                                pass
                src_results.unlink()

            # Clean up noisy.png if created
            noisy_file = self.METHOD_DIR / "color_noisy" / "noisy.png"
            if noisy_file.exists():
                import shutil
                shutil.copy(noisy_file, output_dir / "noisy.png")
                outputs["noisy"] = output_dir / "noisy.png"
                noisy_file.unlink()

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=primary,
                outputs=outputs,
                metrics=metrics
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="QMSANet timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
