"""EMVD Multi-Stage Video Denoising adapter (IPOL preprint 464)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class EMVDVideoDenoisingMethod(IPOLMethod):
    """Efficient multi-stage video denoising using recurrent networks."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_464_video_denoising"

    @property
    def name(self) -> str:
        return "emvd_video_denoising"

    @property
    def display_name(self) -> str:
        return "EMVD Multi-Stage Video Denoising"

    @property
    def description(self) -> str:
        return "Frame-recurrent video denoising with efficient multi-stage network"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DENOISING

    @property
    def input_type(self) -> InputType:
        return InputType.VIDEO

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "iso": {
                "type": "choice",
                "choices": ["3200", "12800"],
                "default": "3200",
                "description": "ISO level for noise model"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run EMVD video denoising using simple inference script."""
        input_path = inputs[0]
        iso = params.get("iso", "3200")

        # Check for checkpoints
        ckpts_dir = self.METHOD_DIR / "ckpts"
        if not ckpts_dir.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="EMVD checkpoints not found in ckpts/ directory"
            )

        # Build command for simple inference
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "simple_inference.py"),
            "--input", str(input_path),
            "--output", str(output_dir),
            "--iso", iso
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )

            outputs = {}
            primary = None

            # Collect output images
            for f in sorted(output_dir.glob("output_*.png")):
                if primary is None:
                    primary = f
                outputs[f.stem] = f

            # Success if we have outputs, regardless of warnings
            if primary:
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=primary,
                    outputs=outputs
                )

            # No outputs - check for actual errors
            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"EMVD denoising failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="EMVD denoising timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
