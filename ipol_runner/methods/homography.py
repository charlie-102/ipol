"""Robust Homography Estimation adapter (IPOL 2023 article 356).

Homography fitting from local affine maps using RANSAC variants.
"""
import subprocess
import sys
import os
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
        return "Estimate homography between images using SIFT and RANSAC"

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
            "max_keypoints": {
                "type": "int",
                "default": 2000,
                "min": 100,
                "max": 10000,
                "description": "Maximum number of SIFT keypoints"
            },
            "ratio_threshold": {
                "type": "float",
                "default": 0.75,
                "min": 0.5,
                "max": 1.0,
                "description": "Lowe's ratio test threshold"
            },
            "ransac_threshold": {
                "type": "float",
                "default": 5.0,
                "min": 1.0,
                "max": 20.0,
                "description": "RANSAC reprojection error threshold (pixels)"
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

        img1 = inputs[0]
        img2 = inputs[1]

        # Use simplified Python script (avoids C++ library and TensorFlow)
        simple_script = self.METHOD_DIR / "homography_simple.py"
        use_simple = simple_script.exists()

        # Check for original C++ library as fallback
        lib_path = self.METHOD_DIR / "libDA.so"
        if not lib_path.exists():
            lib_path = self.METHOD_DIR / "libDA.dylib"

        if not use_simple and not lib_path.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Neither simplified script nor C++ library available"
            )

        # Set environment
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"
        env["OMP_NUM_THREADS"] = "1"

        if use_simple:
            cmd = [
                sys.executable,
                str(simple_script),
                "--im1", str(img1),
                "--im2", str(img2),
                "--workdir", str(output_dir),
                "--max_keypoints", str(params.get("max_keypoints", 2000)),
                "--ratio_threshold", str(params.get("ratio_threshold", 0.75)),
                "--ransac_threshold", str(params.get("ransac_threshold", 5.0)),
            ]
        else:
            # Original script with C++ library
            cmd = [
                sys.executable,
                str(self.METHOD_DIR / "main.py"),
                "--im1", str(img1),
                "--im2", str(img2),
                "--detector", "SIFT",
                "--descriptor", "AID",
                "--affmaps", "locate",
                "--gfilter", "Aff_H_2",
                "--precision", "24.0",
                "--ransac_iters", "1000",
                "--workdir", str(output_dir) + "/"
            ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300,
                env=env
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
