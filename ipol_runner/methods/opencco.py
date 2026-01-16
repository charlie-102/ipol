"""OpenCCO Vascular Tree Generation adapter (IPOL 2023 article 477)."""
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class OpenCCOMethod(IPOLMethod):
    """Generate 2D vascular trees using constrained constructive optimization."""

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

    @property
    def input_count(self) -> int:
        return 0  # Generates vascular trees, no input required

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "num_terminals": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 1000,
                "description": "Number of terminal segments"
            },
            "gamma": {
                "type": "float",
                "default": 3.0,
                "min": 2.1,
                "max": 3.0,
                "description": "Bifurcation exponent (Murray's law)"
            },
            "radius": {
                "type": "float",
                "default": 100.0,
                "min": 10.0,
                "max": 1000.0,
                "description": "Domain radius"
            },
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 999999,
                "description": "Random seed for reproducibility"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run OpenCCO vascular tree generation."""
        num_terminals = params.get("num_terminals", 100)
        gamma = params.get("gamma", 3.0)
        radius = params.get("radius", 100.0)
        seed = params.get("seed", 42)

        # Use simplified Python script (avoids C++ compilation)
        simple_script = self.METHOD_DIR / "opencco_simple.py"
        use_simple = simple_script.exists()

        # Check for compiled binary as fallback
        binary = self.METHOD_DIR / "build" / "bin" / "generateTree2D"

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
                    use_simple = simple_script.exists()
                else:
                    make_result = subprocess.run(
                        ["make", "-j4"],
                        cwd=str(build_dir),
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if make_result.returncode != 0:
                        use_simple = simple_script.exists()

            except Exception:
                use_simple = simple_script.exists()

        if not use_simple and not binary.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Neither Python script nor compiled binary available"
            )

        output_svg = output_dir / "vascular_tree.svg"

        # Set environment
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"

        if use_simple:
            cmd = [
                sys.executable,
                str(simple_script),
                "-n", str(num_terminals),
                "-g", str(gamma),
                "-r", str(radius),
                "-s", str(seed),
                "-o", str(output_svg),
            ]
        else:
            cmd = [
                str(binary),
                "-n", str(num_terminals),
                "-o", str(output_dir / "vascular_tree.vtk"),
            ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600,
                env=env
            )

            outputs = {}
            primary = None

            # Check for outputs
            for pattern in ["*.svg", "*.eps", "*.vtk"]:
                for f in output_dir.glob(pattern):
                    outputs[f.stem] = f
                    if primary is None:
                        primary = f

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
