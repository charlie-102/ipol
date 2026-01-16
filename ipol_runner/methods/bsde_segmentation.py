"""BSDE Image Segmentation adapter (IPOL preprint 636)."""
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class BSDESegmentationMethod(IPOLMethod):
    """Image segmentation using backward stochastic differential equations."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_636_bsde_segmentation"

    @property
    def name(self) -> str:
        return "bsde_segmentation"

    @property
    def display_name(self) -> str:
        return "BSDE Image Segmentation"

    @property
    def description(self) -> str:
        return "Image segmentation using backward stochastic differential equations"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return None  # C++ code, no Python requirements

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "mu": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 100.0,
                "description": "Background parameter mu (0 for automatic, 40-99 for manual)"
            }
        }

    def _ensure_compiled(self) -> bool:
        """Ensure the C++ code is compiled."""
        binary = self.METHOD_DIR / "BSDEseg"
        if binary.exists():
            return True

        try:
            result = subprocess.run(
                ["make"],
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )
            return binary.exists()
        except Exception:
            return False

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run BSDE image segmentation."""
        input_path = inputs[0]

        # Ensure binary is compiled
        if not self._ensure_compiled():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Failed to compile BSDE segmentation. Requires g++ and libpng-dev."
            )

        mu = params.get("mu", 0.0)
        segmented_out = output_dir / "segmented.png"
        binarized_out = output_dir / "binarized.png"
        region_out = output_dir / "region.png"

        cmd = [
            str(self.METHOD_DIR / "BSDEseg"),
            str(input_path),
            str(mu),
            str(segmented_out),
            str(binarized_out),
            str(region_out)
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

            if segmented_out.exists():
                outputs["segmented"] = segmented_out
                primary = segmented_out

            if binarized_out.exists():
                outputs["binarized"] = binarized_out

            if region_out.exists():
                outputs["region"] = region_out

            if not primary:
                error_msg = result.stderr if result.stderr else "No output generated"
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"BSDE segmentation failed: {error_msg}"
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
                error_message="BSDE segmentation timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
