"""SiamTE Camera Trace Extraction adapter (IPOL preprint 558)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SiamTEMethod(IPOLMethod):
    """Camera trace extraction using Siamese network."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_558_siamte"

    @property
    def name(self) -> str:
        return "siamte"

    @property
    def display_name(self) -> str:
        return "SiamTE Camera Trace Extraction"

    @property
    def description(self) -> str:
        return "Deep learning camera trace extraction using Siamese networks"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        req_file = self.METHOD_DIR / "requirements.txt"
        if req_file.exists():
            return req_file
        return None

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {}  # No user-configurable parameters

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run SiamTE camera trace extraction."""
        input_path = inputs[0]

        output_file = output_dir / "output.png"
        residual_file = output_dir / "residual.png"

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            "--input", str(input_path),
            "--output", str(output_file),
            "--residual", str(residual_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600  # Deep learning inference can be slow
            )

            outputs = {}
            primary = None

            if output_file.exists():
                outputs["output"] = output_file
                primary = output_file

            if residual_file.exists():
                outputs["residual"] = residual_file

            # Parse metrics from out.txt if exists
            metrics = {}
            metrics_file = self.METHOD_DIR / "out.txt"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    for line in f:
                        if "Manhattan distance" in line:
                            try:
                                metrics["manhattan_distance"] = float(line.split(":")[-1].strip())
                            except ValueError:
                                pass
                        elif "NIQE" in line:
                            try:
                                key = "niqe_output" if "output" in line else "niqe_input"
                                metrics[key] = float(line.split(":")[-1].strip())
                            except ValueError:
                                pass
                metrics_file.unlink()

            if not primary:
                error_msg = result.stderr if result.stderr else "No output generated"
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"SiamTE failed: {error_msg}"
                )

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=primary,
                outputs=outputs,
                metrics=metrics
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="SiamTE timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
