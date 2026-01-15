"""Semiogram Gait Analysis adapter (IPOL 535)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SemiogramMethod(IPOLMethod):
    """Semiogram - Gait quantification using IMU sensor data."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_535_semiogram"

    @property
    def name(self) -> str:
        return "semiogram"

    @property
    def display_name(self) -> str:
        return "Semiogram (Gait Analysis)"

    @property
    def description(self) -> str:
        return "Quantify gait parameters from IMU sensor data"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.MEDICAL

    @property
    def input_type(self) -> InputType:
        return InputType.SENSOR_DATA

    @property
    def input_count(self) -> int:
        return 2  # sensor data + metadata

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "freq": {
                "type": "int",
                "default": 100,  # Common IMU sampling rate
                "description": "Acquisition frequency (Hz)"
            },
            "distance": {
                "type": "int",
                "default": 10,  # Standard 10m walking test
                "description": "Walked distance in meters"
            },
            "min_z": {
                "type": "int",
                "default": -3,
                "description": "Minimum Z-score for visualization"
            },
            "max_z": {
                "type": "int",
                "default": 3,
                "description": "Maximum Z-score for visualization"
            }
        }

    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        # freq and distance now have defaults
        return None

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Requires 2 inputs: sensor data file and metadata file"
            )

        sensor_data = inputs[0]
        metadata = inputs[1]

        cmd = [
            sys.executable, str(self.METHOD_DIR / "main.py"),
            "-i0", str(sensor_data),
            "-i1", str(metadata),
            "-freq", str(int(params.get("freq", 100))),
            "-distance", str(int(params.get("distance", 10))),
            "-min_z", str(int(params.get("min_z", -3))),
            "-max_z", str(int(params.get("max_z", 3))),
        ]

        # Add reference inputs if provided (inputs 2 and 3)
        if len(inputs) >= 4:
            cmd.extend(["-i2", str(inputs[2]), "-i3", str(inputs[3])])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(output_dir),
                capture_output=True,
                text=True,
                timeout=120
            )

            outputs = {}
            primary = None

            # Check for outputs
            semio_svg = output_dir / "semio.svg"
            if semio_svg.exists():
                outputs["semiogram"] = semio_svg
                primary = semio_svg

            params_txt = output_dir / "trial_parameters.txt"
            if params_txt.exists():
                outputs["parameters"] = params_txt

            criteria_txt = output_dir / "trial_criteria.txt"
            if criteria_txt.exists():
                outputs["criteria"] = criteria_txt

            if not primary:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stderr: {result.stderr}"
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
                error_message="Semiogram timed out after 120s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
