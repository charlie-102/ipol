"""Voronoi Page Segmentation adapter (IPOL preprint 591)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class VoronoiSegmentationMethod(IPOLMethod):
    """Voronoi diagrams for document page segmentation."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_591_voronoi_segmentation"

    @property
    def name(self) -> str:
        return "voronoi_segmentation"

    @property
    def display_name(self) -> str:
        return "Voronoi Page Segmentation"

    @property
    def description(self) -> str:
        return "Document layout analysis using area Voronoi diagrams"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "binarization_method": {
                "type": "choice",
                "choices": ["otsu", "threshold", "local"],
                "default": "otsu",
                "description": "Binarization method for non-binary input images"
            },
            "subsample_method": {
                "type": "choice",
                "choices": ["random", "grid"],
                "default": "random",
                "description": "Border subsampling method"
            },
            "subsample_param": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "description": "Subsampling parameter (probability for random, spacing for grid)"
            },
            "remove_blobs": {
                "type": "int",
                "default": 4,
                "min": 0,
                "max": 100,
                "description": "Remove blobs smaller than this size (0 to disable)"
            },
            "save_all_stages": {
                "type": "bool",
                "default": False,
                "description": "Save intermediate processing stages"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run Voronoi page segmentation."""
        input_path = inputs[0]

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "voronoi.py"),
            "-i", str(input_path),
            "-o", str(output_dir) + "/voronoi_",
            "-B", params.get("binarization_method", "otsu"),
            "-s", params.get("subsample_method", "random"),
            "-r", str(params.get("subsample_param", 0.1)),
            "-b", str(params.get("remove_blobs", 4)),
        ]

        if params.get("save_all_stages", False):
            cmd.extend(["-S", "all"])
        else:
            cmd.extend(["-S", "result"])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check for expected outputs
            outputs = {}
            primary = None

            # Final result is 9_final_area_voronoi.png
            final_output = output_dir / "voronoi_9_final_area_voronoi.png"
            if final_output.exists():
                outputs["segmentation"] = final_output
                primary = final_output

            # Also capture intermediate stages if requested
            if params.get("save_all_stages", False):
                for stage in ["1_binarized", "6_pruned_redundant", "8_pruned_by_features"]:
                    stage_file = output_dir / f"voronoi_{stage}.png"
                    if stage_file.exists():
                        outputs[stage] = stage_file

            if not primary:
                # Check if there was an error message in output
                error_msg = result.stderr if result.stderr else "No output generated"
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Voronoi segmentation failed: {error_msg}"
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
                error_message="Voronoi segmentation timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
