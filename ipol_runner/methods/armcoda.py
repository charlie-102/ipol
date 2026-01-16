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

        # This method generates an interactive Plotly animation
        # We'll run it and capture the HTML output
        cmd = [
            sys.executable, "-c",
            f"""
import sys
sys.path.insert(0, '{self.METHOD_DIR}')
from main import main
main({subject}, '{movement}', {sensor})
"""
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"ARMCODA failed: {result.stderr}"
                )

            # The method creates visualization files
            # Check for any generated outputs
            outputs = {}
            primary = None

            for out_file in self.METHOD_DIR.glob("*.html"):
                import shutil
                dst = output_dir / out_file.name
                shutil.copy(out_file, dst)
                outputs[out_file.stem] = dst
                if primary is None:
                    primary = dst

            for out_file in self.METHOD_DIR.glob("*.png"):
                import shutil
                dst = output_dir / out_file.name
                shutil.copy(out_file, dst)
                outputs[out_file.stem] = dst
                if primary is None:
                    primary = dst

            if not primary:
                # Create a simple info file if no visual output
                info_file = output_dir / "analysis_info.txt"
                info_file.write_text(f"ARMCODA Analysis\nSubject: {subject}\nMovement: {movement}\nSensor: {sensor}\n\n{result.stdout}")
                outputs["info"] = info_file
                primary = info_file

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
