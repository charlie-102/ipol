"""Handheld Burst Super-Resolution adapter (IPOL 2023 article 460)."""
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class BurstSuperResMethod(IPOLMethod):
    """Handheld burst super-resolution from multiple frames."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_460_burst_superres"

    @property
    def name(self) -> str:
        return "burst_superres"

    @property
    def display_name(self) -> str:
        return "Handheld Burst Super-Resolution"

    @property
    def description(self) -> str:
        return "Enhance resolution from multiple handheld camera frames"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.VIDEO  # Expects directory of image files

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "scale": {
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 4.0,
                "description": "Upscaling factor"
            },
            "robustness": {
                "type": "bool",
                "default": False,
                "description": "Enable robustness mode for moving scenes"
            },
            "post_process": {
                "type": "bool",
                "default": False,
                "description": "Apply post-processing (sharpening, tonemapping)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run burst super-resolution."""
        input_path = inputs[0]  # Directory containing image files

        # Check for image files (DNG or standard formats)
        dng_files = list(input_path.glob("*.dng")) + list(input_path.glob("*.DNG"))
        img_files = (list(input_path.glob("*.png")) + list(input_path.glob("*.PNG")) +
                     list(input_path.glob("*.jpg")) + list(input_path.glob("*.JPG")) +
                     list(input_path.glob("*.jpeg")) + list(input_path.glob("*.JPEG")))

        use_simplified = False
        if dng_files:
            # Check if all dependencies for original method are available
            try:
                import rawpy
                import polyblur  # Required by original demo.py
                import colour_demosaicing
            except ImportError:
                use_simplified = True
        elif img_files:
            use_simplified = True
        else:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="No image files found. Provide a directory with DNG, PNG, or JPEG files."
            )

        output_path = output_dir / "super_resolved.png"

        # Set environment for cache directories
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = "/tmp/claude/matplotlib"
        env["TORCH_HOME"] = "/tmp/claude/torch_cache"
        env["TMPDIR"] = "/tmp/claude"
        env["OMP_NUM_THREADS"] = "1"
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Ensure cache directories exist
        for d in ["/tmp/claude/matplotlib", "/tmp/claude/torch_cache"]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Select script based on input type
        if use_simplified:
            script = self.METHOD_DIR / "demo_simple.py"
            if not script.exists():
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message="Simplified script not found and rawpy not installed for DNG support."
                )
        else:
            script = self.METHOD_DIR / "demo.py"

        cmd = [
            sys.executable,
            str(script),
            "--impath", str(input_path),
            "--outpath", str(output_path),
            "--scale", str(params.get("scale", 2.0)),
            "--R_on", str(params.get("robustness", False)),
            "--post_process", str(params.get("post_process", False)),
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600,  # Longer timeout for burst processing
                env=env
            )

            if output_path.exists():
                outputs = {"super_resolved": output_path}
                # Check for additional outputs
                for pattern in ["*_ref.png", "*_robustness.png", "input_0*.png", "robustness_map*.png"]:
                    for f in output_dir.glob(pattern):
                        outputs[f.stem] = f

                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_path,
                    outputs=outputs
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Burst super-resolution failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
