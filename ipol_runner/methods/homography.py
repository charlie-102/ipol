"""Robust Homography Estimation adapter (IPOL 2023 article 356).

Homography fitting from local affine maps using RANSAC variants.
"""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class HomographyMethod(IPOLMethod):
    """Robust homography estimation from local affine maps."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_356_homography"

    @property
    def name(self) -> str:
        return "homography"

    @property
    def display_name(self) -> str:
        return "Robust Homography Estimation"

    @property
    def description(self) -> str:
        return "Estimate homography between images using affine RANSAC methods"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION  # Feature matching/detection

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR

    @property
    def input_count(self) -> int:
        return 2

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "detector": {
                "type": "choice",
                "choices": ["SIFT", "HessAff"],
                "default": "SIFT",
                "description": "Keypoint detector"
            },
            "descriptor": {
                "type": "choice",
                "choices": ["AID", "HardNet"],
                "default": "AID",
                "description": "Feature descriptor"
            },
            "affmaps": {
                "type": "choice",
                "choices": ["locate", "affnet", "simple"],
                "default": "locate",
                "description": "Affine map estimation method"
            },
            "gfilter": {
                "type": "choice",
                "choices": ["Aff_H_0", "Aff_H_1", "Aff_H_2", "Aff_O_0", "Aff_O_1", "Aff_O_2"],
                "default": "Aff_H_2",
                "description": "Geometric filter (RANSAC variant)"
            },
            "precision": {
                "type": "float",
                "default": 24.0,
                "min": 1.0,
                "max": 100.0,
                "description": "Precision of symmetric transfer error"
            },
            "ransac_iters": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "description": "Number of RANSAC iterations"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run homography estimation."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Two images required for homography estimation"
            )

        # Check for compiled library
        lib_path = self.METHOD_DIR / "libDA.so"
        if not lib_path.exists():
            # Try .dylib on macOS
            lib_path = self.METHOD_DIR / "libDA.dylib"
        if not lib_path.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="C++ library not compiled. Run: cd methods/ipol_2023_356_homography && mkdir build && cd build && cmake .. && make && mv libDA.so .."
            )

        img1 = inputs[0]
        img2 = inputs[1]

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "main.py"),
            "--im1", str(img1),
            "--im2", str(img2),
            "--detector", params.get("detector", "SIFT"),
            "--descriptor", params.get("descriptor", "AID"),
            "--affmaps", params.get("affmaps", "locate"),
            "--gfilter", params.get("gfilter", "Aff_H_2"),
            "--precision", str(params.get("precision", 24.0)),
            "--ransac_iters", str(params.get("ransac_iters", 1000)),
            "--workdir", str(output_dir) + "/"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Check for outputs
            outputs = {}
            primary = None

            matches_file = output_dir / "all_matches.png"
            warped_file = output_dir / "queryontarget.png"

            if matches_file.exists():
                outputs["matches"] = matches_file
                primary = matches_file

            if warped_file.exists():
                outputs["warped"] = warped_file

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
                error_message=f"Homography estimation failed: {error_msg[:500]}"
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
