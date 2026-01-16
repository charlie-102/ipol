"""Signal Decomposition adapter (IPOL 2023 article 417).

Python implementation of the ADMM-based signal decomposition.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SignalDecompositionMethod(IPOLMethod):
    """Two-stage signal decomposition into Jump, Oscillation and Trend using ADMM."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_417_signal_decomposition"

    @property
    def name(self) -> str:
        return "signal_decomposition"

    @property
    def display_name(self) -> str:
        return "Signal Decomposition (JOT)"

    @property
    def description(self) -> str:
        return "Decompose 1D signals into Jump, Oscillation and Trend components"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION  # Signal analysis

    @property
    def input_type(self) -> InputType:
        return InputType.SENSOR_DATA  # 1D signal data in CSV format

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "a_bar": {
                "type": "float",
                "default": 0.3,
                "min": 0.1,
                "max": 1.0,
                "description": "Minimal expected discontinuity/jump height"
            },
            "tau": {
                "type": "float",
                "default": 10.0,
                "min": 1.0,
                "max": 100.0,
                "description": "Factor for beta parameter"
            },
            "stage2": {
                "type": "bool",
                "default": False,
                "description": "Enable Stage 2 refinement"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run signal decomposition using Python implementation."""
        input_path = inputs[0]

        # Verify CSV input
        if not input_path.suffix.lower() == ".csv":
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Input must be a CSV file containing 1D signal data"
            )

        output_file = output_dir / "decomposition.csv"

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "signal_decomposition.py"),
            str(input_path),
            str(output_file),
            "--a-bar", str(params.get("a_bar", 0.3)),
            "--tau", str(params.get("tau", 10.0)),
        ]

        if params.get("stage2", False):
            cmd.append("--stage2")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            if output_file.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_file,
                    outputs={"decomposition": output_file}
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Signal decomposition failed: {error_msg[:500]}"
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
