"""OpenCCO Vascular Tree Generation adapter (IPOL 2023 article 477)."""
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class OpenCCOMethod(IPOLMethod):
    """Generate 2D and 3D vascular trees using constrained constructive optimization."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_477_opencco"

    @property
    def name(self) -> str:
        return "opencco"

    @property
    def display_name(self) -> str:
        return "OpenCCO Vascular Tree Generation"

    @property
    def description(self) -> str:
        return "Generate realistic vascular network structures using CCO algorithm"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.GENERATION

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID  # Uses predefined parameters, not input files

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "mode": {
                "type": "choice",
                "choices": ["2D", "3D"],
                "default": "2D",
                "description": "Generate 2D or 3D vascular tree"
            },
            "num_terminals": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 10000,
                "description": "Number of terminal segments"
            },
            "perfusion_volume": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 100.0,
                "description": "Perfusion volume parameter"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run OpenCCO vascular tree generation."""
        mode = params.get("mode", "2D")
        num_terminals = params.get("num_terminals", 100)

        # Check for compiled binary
        binary_name = f"generateTree{mode}"
        binary = self.METHOD_DIR / "build" / "bin" / binary_name

        if not binary.exists():
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
                    timeout=300
                )

                if make_result.returncode != 0:
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"Compilation failed: {make_result.stderr[:300]}"
                    )

            except FileNotFoundError:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message="CMake not found. Install cmake to compile OpenCCO."
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

        output_file = output_dir / f"vascular_tree_{mode}.vtk"

        cmd = [
            str(binary),
            "-o", str(output_file),
            "-n", str(num_terminals),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )

            if output_file.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_file,
                    outputs={"vtk": output_file}
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"OpenCCO failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
