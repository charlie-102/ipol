"""Binary Shape Vectorization adapter (IPOL 2023 article 401)."""
import subprocess
import sys
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
        return "Convert binary shapes to vector representations using affine scale-space"

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
                "default": 1.0,
                "min": 0.1,
                "max": 10.0,
                "description": "Scale parameter for vectorization"
            },
            "precision": {
                "type": "float",
                "default": 0.5,
                "min": 0.1,
                "max": 5.0,
                "description": "Precision for curve approximation"
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

        # Check for compiled binary
        binary = self.METHOD_DIR / "affine_sp_vectorization"
        if not binary.exists():
            # Try to compile
            try:
                # Create build directory
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
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"CMake failed: {cmake_result.stderr[:300]}"
                    )

                make_result = subprocess.run(
                    ["make", "-j4"],
                    cwd=str(build_dir),
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if make_result.returncode != 0:
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"Compilation failed: {make_result.stderr[:300]}"
                    )

                # Move binary to METHOD_DIR
                built_binary = build_dir / "affine_sp_vectorization"
                if built_binary.exists():
                    import shutil
                    shutil.copy(str(built_binary), str(binary))

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
                error_message="Binary not found and compilation failed"
            )

        output_svg = output_dir / "vectorized.svg"
        output_png = output_dir / "output.png"

        cmd = [
            str(binary),
            str(input_path),
            str(output_svg),
            "-s", str(params.get("scale", 1.0)),
            "-p", str(params.get("precision", 0.5)),
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

            if output_svg.exists():
                primary = output_svg
                outputs["svg"] = output_svg

            if output_png.exists():
                outputs["png"] = output_png

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
