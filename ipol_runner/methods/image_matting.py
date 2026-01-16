"""Natural Image Matting adapter (IPOL preprint 532)."""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class ImageMattingMethod(IPOLMethod):
    """Natural image matting using closed-form solution."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_532_image_matting" / "code"

    @property
    def name(self) -> str:
        return "image_matting"

    @property
    def display_name(self) -> str:
        return "Natural Image Matting"

    @property
    def description(self) -> str:
        return "Closed-form solution to natural image matting"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR

    @property
    def input_count(self) -> int:
        return 2  # image + trimap/scribbles

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
        """Run natural image matting."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Image matting requires 2 inputs: image and trimap/scribbles"
            )

        image_path = inputs[0]
        scribbles_path = inputs[1]

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            str(image_path),
            "-s", str(scribbles_path)
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

            # The script outputs to current directory
            src_output = self.METHOD_DIR / "output.png"
            src_input = self.METHOD_DIR / "input.png"
            src_scribbles = self.METHOD_DIR / "scribles.png"

            if src_output.exists():
                dst = output_dir / "alpha_matte.png"
                shutil.copy(src_output, dst)
                outputs["alpha_matte"] = dst
                primary = dst
                src_output.unlink()

            if src_input.exists():
                dst = output_dir / "input.png"
                shutil.copy(src_input, dst)
                outputs["input"] = dst
                src_input.unlink()

            if src_scribbles.exists():
                dst = output_dir / "scribbles.png"
                shutil.copy(src_scribbles, dst)
                outputs["scribbles"] = dst
                src_scribbles.unlink()

            if not primary:
                error_msg = result.stderr if result.stderr else "No output generated"
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Image matting failed: {error_msg}"
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
                error_message="Image matting timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
