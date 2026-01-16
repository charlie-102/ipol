"""BSDE Image Denoising adapter (IPOL 2023 article 467)."""
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class BSDEDenoisingMethod(IPOLMethod):
    """Image denoising based on Backward Stochastic Differential Equations."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_467_bsde_denoising"

    @property
    def name(self) -> str:
        return "bsde_denoising"

    @property
    def display_name(self) -> str:
        return "BSDE Image Denoising"

    @property
    def description(self) -> str:
        return "Denoise images using Backward Stochastic Differential Equations"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "sigma": {
                "type": "float",
                "default": 20.0,
                "min": 1.0,
                "max": 100.0,
                "description": "Noise standard deviation"
            },
            "add_noise": {
                "type": "bool",
                "default": True,
                "description": "Add synthetic noise to input (False if already noisy)"
            },
            "enhance": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "description": "Enhancing parameter c (0=no enhancement)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run BSDE denoising."""
        input_path = inputs[0]

        # Check for compiled binary
        binary = self.METHOD_DIR / "BSDE"
        if not binary.exists():
            # Try to compile
            try:
                make_result = subprocess.run(
                    ["make"],
                    cwd=str(self.METHOD_DIR),
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if make_result.returncode != 0:
                    # Check for common issues
                    if "openmp" in make_result.stderr.lower() or "omp.h" in make_result.stderr.lower():
                        return MethodResult(
                            success=False,
                            output_dir=output_dir,
                            error_message="OpenMP not available. On macOS, use: brew install libomp"
                        )
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"Compilation failed: {make_result.stderr[:300]}"
                    )

            except FileNotFoundError:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message="make/g++ not found. Install build tools."
                )
            except Exception as e:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Compilation error: {str(e)}"
                )

        if not binary.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Binary not found after compilation attempt"
            )

        noisy_path = output_dir / "noisy.png"
        denoised_path = output_dir / "denoised.png"

        sigma = params.get("sigma", 20.0)
        add_noise = 1 if params.get("add_noise", True) else 0
        enhance = params.get("enhance", 0.0)

        # BSDE original.png sigma noisy.png denoised.png add_noise c
        cmd = [
            str(binary),
            str(input_path),
            str(sigma),
            str(noisy_path),
            str(denoised_path),
            str(add_noise),
            str(enhance)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            outputs = {}
            primary = None

            if denoised_path.exists():
                primary = denoised_path
                outputs["denoised"] = denoised_path

            if noisy_path.exists():
                outputs["noisy"] = noisy_path

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
                error_message=f"BSDE denoising failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
