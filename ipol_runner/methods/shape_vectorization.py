"""Binary Shape Vectorization adapter (IPOL 2023 article 401)."""
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class ShapeVectorizationMethod(IPOLMethod):
    """Binary shape vectorization using affine scale-space."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_401_shape_vectorization"

    @property
    def name(self) -> str:
        return "shape_vectorization"

    @property
    def display_name(self) -> str:
        return "Binary Shape Vectorization"

    @property
    def description(self) -> str:
        return "Convert binary shapes to vector representations"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "scale": {
                "type": "float",
                "default": 2.0,
                "min": 0.1,
                "max": 10.0,
                "description": "Smoothing scale parameter"
            },
            "threshold": {
                "type": "float",
                "default": 127.5,
                "min": 0,
                "max": 255,
                "description": "Binarization threshold"
            },
            "error_threshold": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "description": "Bézier approximation error threshold (pixels)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run binary shape vectorization."""
        input_path = inputs[0]

        # Use simplified Python script (avoids C++ compilation)
        simple_script = self.METHOD_DIR / "vectorize_simple.py"
        use_simple = simple_script.exists()

        # Check for compiled binary as fallback
        binary = self.METHOD_DIR / "affine_sp_vectorization"
        if not use_simple and not binary.exists():
            # Try to compile
            try:
                build_dir = self.METHOD_DIR / "build"
                build_dir.mkdir(exist_ok=True)

                cmake_result = subprocess.run(
                    ["cmake", ".."],
                    cwd=str(build_dir),
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if cmake_result.returncode != 0:
                    # Fall back to Python script
                    use_simple = simple_script.exists()
                else:
                    make_result = subprocess.run(
                        ["make", "-j4"],
                        cwd=str(build_dir),
                        capture_output=True,
                        text=True,
                        timeout=120
                    )

                    if make_result.returncode != 0:
                        use_simple = simple_script.exists()
                    else:
                        built_binary = build_dir / "affine_sp_vectorization"
                        if built_binary.exists():
                            import shutil
                            shutil.copy(str(built_binary), str(binary))

            except Exception:
                use_simple = simple_script.exists()

        if not use_simple and not binary.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Neither Python script nor compiled binary available"
            )

        output_svg = output_dir / "vectorized.svg"

        # Set environment
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"

        if use_simple:
            # Use Python script
            cmd = [
                sys.executable,
                str(simple_script),
                str(input_path),
                "-s", str(params.get("scale", 2.0)),
                "-f", str(params.get("threshold", 127.5)),
                "-T", str(params.get("error_threshold", 1.0)),
                "-O", str(output_svg),
            ]
        else:
            # Use C++ binary
            cmd = [
                str(binary),
                str(input_path),
                "-s", str(params.get("scale", 2.0)),
                "-f", str(params.get("threshold", 127.5)),
                "-T", str(params.get("error_threshold", 1.0)),
                "-O", str(output_svg),
            ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            outputs = {}
            primary = None

            if output_svg.exists():
                primary = output_svg
                outputs["svg"] = output_svg

            # Check for any PNG outputs
            for png in output_dir.glob("*.png"):
                outputs[png.stem] = png

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
                error_message=f"Vectorization failed: {error_msg[:500]}"
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
