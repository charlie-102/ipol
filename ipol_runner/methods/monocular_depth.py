"""Monocular Depth Estimation adapter (IPOL 2023 article 459).

Review of state-of-the-art monocular depth methods: MiDaS, DPT, Adabins, GLPDepth.
"""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class MonocularDepthMethod(IPOLMethod):
    """Monocular depth estimation using multiple architectures."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_459_monocular_depth"

    @property
    def name(self) -> str:
        return "monocular_depth"

    @property
    def display_name(self) -> str:
        return "Monocular Depth Estimation"

    @property
    def description(self) -> str:
        return "Depth estimation from single images (MiDaS, DPT, Adabins, GLPDepth)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "method": {
                "type": "choice",
                "choices": ["midas", "dpt", "adabins", "glpdepth"],
                "default": "midas",
                "description": "Depth estimation method"
            },
            "model_type": {
                "type": "choice",
                "choices": ["dpt_large", "dpt_hybrid", "midas_v21", "midas_v21_small"],
                "default": "midas_v21_small",
                "description": "Model variant (for MiDaS/DPT)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run monocular depth estimation."""
        input_path = inputs[0]
        method = params.get("method", "midas")
        model_type = params.get("model_type", "midas_v21_small")

        # Map method to subdirectory
        method_dirs = {
            "midas": "MiDaS-main",
            "dpt": "DPT-master",
            "adabins": "Adabins-main",
            "glpdepth": "GLPDepth-main"
        }

        method_dir = self.METHOD_DIR / method_dirs.get(method, "MiDaS-main")
        if not method_dir.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Method directory not found: {method_dir}"
            )

        # Create temp input directory
        temp_input = output_dir / "input"
        temp_input.mkdir(exist_ok=True)
        shutil.copy(str(input_path), str(temp_input / input_path.name))

        try:
            if method in ["midas", "dpt"]:
                # MiDaS and DPT share similar structure
                script = method_dir / "midas" / "run.py"

                # Check for model weights
                weights_dir = method_dir / "weights"
                model_file = weights_dir / f"{model_type}.pt"
                if not model_file.exists():
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"Model weights not found: {model_file}. Download from IPOL."
                    )

                cmd = [
                    sys.executable,
                    str(script),
                    "--input_path", str(temp_input),
                    "--output_path", str(output_dir),
                    "--model_weights", str(model_file),
                    "--model_type", model_type
                ]

            elif method == "adabins":
                script = method_dir / "adabin" / "infer.py"
                cmd = [
                    sys.executable,
                    str(script),
                    "--image", str(input_path),
                    "--output", str(output_dir / "depth.png")
                ]

            elif method == "glpdepth":
                script = method_dir / "glpdepth" / "test.py"
                cmd = [
                    sys.executable,
                    str(script),
                    "--image", str(input_path),
                    "--output", str(output_dir / "depth.png")
                ]

            result = subprocess.run(
                cmd,
                cwd=str(method_dir),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Find output files
            outputs = {}
            primary = None

            # Check common output patterns
            for pattern in ["*.png", "*.pfm"]:
                for f in output_dir.glob(pattern):
                    if f.name != input_path.name:
                        outputs[f.stem] = f
                        if primary is None:
                            primary = f

            # Cleanup temp
            shutil.rmtree(temp_input, ignore_errors=True)

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
                error_message=f"Depth estimation failed: {error_msg[:500]}"
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
        finally:
            shutil.rmtree(temp_input, ignore_errors=True)
