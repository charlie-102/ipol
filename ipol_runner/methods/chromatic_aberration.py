"""Chromatic Aberration Correction adapter (IPOL 2023 article 443)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class ChromaticAberrationMethod(IPOLMethod):
    """Fast chromatic aberration correction with 1D filters."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_443_chromatic_aberration"

    @property
    def name(self) -> str:
        return "chromatic_aberration"

    @property
    def display_name(self) -> str:
        return "Chromatic Aberration Correction"

    @property
    def description(self) -> str:
        return "Remove color fringing in images using efficient 1D filtering"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "L_hor": {
                "type": "int",
                "default": 7,
                "min": 1,
                "max": 15,
                "description": "Horizontal filter length"
            },
            "L_ver": {
                "type": "int",
                "default": 4,
                "min": 1,
                "max": 15,
                "description": "Vertical filter length"
            },
            "alpha_R": {
                "type": "float",
                "default": 0.5,
                "description": "Red channel alpha parameter"
            },
            "alpha_B": {
                "type": "float",
                "default": 1.0,
                "description": "Blue channel alpha parameter"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run chromatic aberration correction."""
        input_path = inputs[0]
        output_path = output_dir / "corrected.png"

        # Check if Cython module needs compilation
        cython_so = list(self.METHOD_DIR.glob("filter_cython*.so"))
        if not cython_so:
            # Try to compile
            try:
                compile_result = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=str(self.METHOD_DIR),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if compile_result.returncode != 0:
                    return MethodResult(
                        success=False,
                        output_dir=output_dir,
                        error_message=f"Failed to compile Cython module: {compile_result.stderr[:500]}"
                    )
            except Exception as e:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Cython compilation error: {str(e)}"
                )

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            "--input", str(input_path),
            "--output", str(output_path),
            "--L_hor", str(params.get("L_hor", 7)),
            "--L_ver", str(params.get("L_ver", 4)),
            "--alpha_R", str(params.get("alpha_R", 0.5)),
            "--alpha_B", str(params.get("alpha_B", 1.0)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            if output_path.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_path,
                    outputs={"corrected": output_path}
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Chromatic aberration correction failed: {error_msg[:500]}"
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
