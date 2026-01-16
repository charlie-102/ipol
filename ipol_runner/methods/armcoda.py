"""ARMCODA upper-limb movement analysis adapter (IPOL 2024 article 494)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class ARMCODAMethod(IPOLMethod):
    """ARMCODA upper-limb movement coordination analysis."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_494_armcoda"

    # Pre-defined subjects and movements
    SUBJECTS = list(range(1, 11))  # 1-10
    MOVEMENTS = ["gesture1", "gesture2", "gesture3", "gesture4", "gesture5"]

    @property
    def name(self) -> str:
        return "armcoda"

    @property
    def display_name(self) -> str:
        return "ARMCODA Movement Analysis"

    @property
    def description(self) -> str:
        return "Analyze upper-limb movement coordination from motion capture data"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.MEDICAL

    @property
    def input_type(self) -> InputType:
        return InputType.SENSOR_DATA

    @property
    def input_count(self) -> int:
        return 0  # Uses internal motion capture dataset, no file input required

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "subject": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 10,
                "description": "Subject ID (1-10)"
            },
            "movement": {
                "type": "choice",
                "choices": self.MOVEMENTS,
                "default": "gesture1",
                "description": "Movement type to analyze"
            },
            "sensor": {
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 33,
                "description": "Sensor index to highlight (0-33)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run ARMCODA movement analysis."""
        # Get parameters
        subject = params.get("subject", 1)
        movement = params.get("movement", "gesture1")
        sensor = params.get("sensor", 0)

        # Use the core script that doesn't require plotly
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "armcoda_core.py"),
            "--subject", str(subject),
            "--movement", movement,
            "--sensor", str(sensor),
            "--output", str(output_dir)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"ARMCODA failed: {result.stderr}"
                )

            # Check for generated outputs
            outputs = {}
            primary = None

            # Look for analysis summary
            summary_file = output_dir / "analysis_summary.txt"
            if summary_file.exists():
                outputs["summary"] = summary_file
                primary = summary_file

            # Look for stats JSON
            stats_file = output_dir / "movement_stats.json"
            if stats_file.exists():
                outputs["stats"] = stats_file
                if primary is None:
                    primary = stats_file

            # Look for numpy files
            for npy_file in output_dir.glob("*.npy"):
                outputs[npy_file.stem] = npy_file

            if not primary:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stdout: {result.stdout}, stderr: {result.stderr}"
                )

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=primary,
                outputs=outputs
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="ARMCODA timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
