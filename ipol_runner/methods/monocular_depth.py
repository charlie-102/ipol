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
                "choices": ["midas", "dpt", "glpdepth", "adabins"],
                "default": "midas",
                "description": "Depth estimation method"
            },
            "model_type": {
                "type": "choice",
                "choices": ["midas_v21_small", "midas_v21", "dpt_hybrid", "dpt_large", "nyu", "kitti"],
                "default": "midas_v21_small",
                "description": "Model variant (midas/dpt for MiDaS, nyu/kitti for GLPDepth)"
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

        method_dir = self.METHOD_DIR / method_dirs.get(method, "GLPDepth-main")
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

        import os
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"
        env["TORCH_HOME"] = "/tmp/claude/torch_cache"
        env["TMPDIR"] = "/tmp/claude"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid macOS OpenMP conflict

        try:
            if method == "glpdepth":
                # GLPDepth - recommended, works offline
                glpdepth_dir = method_dir / "glpdepth"
                script = glpdepth_dir / "test.py"

                # Select checkpoint based on model_type
                if model_type in ["nyu", "kitti"]:
                    ckpt_file = glpdepth_dir / f"best_model_{model_type}.ckpt"
                else:
                    ckpt_file = glpdepth_dir / "best_model_nyu.ckpt"

                if not ckpt_file.exists():
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"GLPDepth checkpoint not found: {ckpt_file}. Download from IPOL."
                    )

                # GLPDepth expects a directory with images for --data_path
                # and outputs to --result_dir/--exp_name/
                cmd = [
                    sys.executable,
                    str(script),
                    "--dataset", "imagepath",
                    "--data_path", str(temp_input),
                    "--ckpt_dir", str(ckpt_file),
                    "--result_dir", str(output_dir),
                    "--exp_name", "depth_output",
                    "--gpu_or_cpu", "cpu",
                    "--save_visualize"
                ]

                result = subprocess.run(
                    cmd,
                    cwd=str(glpdepth_dir),
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env
                )

                # GLPDepth saves to result_dir/exp_name/filename
                glp_output_dir = output_dir / "depth_output"

            elif method in ["midas", "dpt"]:
                # Use simplified script (Laplacian-based) to avoid timm version incompatibility
                # The original MiDaS checkpoint requires older timm version
                script = method_dir / "midas" / "run_small.py"
                use_simplified = script.exists()
                if not use_simplified:
                    script = method_dir / "midas" / "run.py"

                cmd = [
                    sys.executable,
                    str(script),
                    "--input_path", str(temp_input),
                    "--output_path", str(output_dir),
                ]

                # Only check/pass model weights if using original script
                if not use_simplified:
                    weights_dir = method_dir / "weights"
                    weights_dir.mkdir(exist_ok=True)
                    model_file = weights_dir / f"{model_type}.pt"

                    if not model_file.exists() or model_file.stat().st_size < 1000:
                        return MethodResult(
                            success=False,
                            output_dir=output_dir,
                            error_message=f"Model weights not found: {model_file}. Download from IPOL."
                        )

                    cmd.extend(["--model_weights", str(model_file), "--model_type", model_type])

                result = subprocess.run(
                    cmd,
                    cwd=str(method_dir),
                    capture_output=True,
                    text=True,
                    timeout=300,
                    env=env
                )

            elif method == "adabins":
                script = method_dir / "adabin" / "infer.py"
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
                    timeout=300,
                    env=env
                )

            # Find output files
            outputs = {}
            primary = None

            # For GLPDepth, check the subdirectory
            if method == "glpdepth":
                glp_output_dir = output_dir / "depth_output"
                search_dirs = [glp_output_dir, output_dir]
            else:
                search_dirs = [output_dir]

            # Check common output patterns
            # For depth estimation, output file may have same name as input (but in output dir)
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for pattern in ["*.png", "*.pfm"]:
                    for f in search_dir.glob(pattern):
                        # Skip files in input directory, but accept same-named files in output
                        if f.parent != temp_input:
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
