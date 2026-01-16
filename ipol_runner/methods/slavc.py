"""SLAVC Sound Source Localization adapter (IPOL 2024 article 525).

Visual sound source localization using audio-visual features.
"""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SLAVCMethod(IPOLMethod):
    """Sound source localization in images using audio-visual learning.

    Takes an image and audio file, localizes the sound source in the image.
    Based on SLAVC (Self-supervised Audio-Visual Learning for Sound Localization).
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_525_slavc"
    WEIGHTS_URL = "https://kiwi.cmla.ens-cachan.fr/index.php/s/yLT6TiyiwXGB54t/download?path=%2F77777000453&files=latest.pth"

    @property
    def name(self) -> str:
        return "slavc"

    @property
    def display_name(self) -> str:
        return "SLAVC Sound Localization"

    @property
    def description(self) -> str:
        return "Visual sound source localization using audio-visual deep learning"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE  # Primary input is image, audio is secondary

    @property
    def input_count(self) -> int:
        return 2  # image and audio file

    @property
    def requires_cuda(self) -> bool:
        return True  # Requires GPU for deep learning

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "alpha": {
                "type": "float",
                "default": 0.4,
                "min": 0.0,
                "max": 1.0,
                "description": "Weight for audio-visual vs object saliency (0=object only, 1=audio-visual only)"
            }
        }

    def _ensure_weights(self) -> tuple[bool, str]:
        """Ensure model weights are downloaded."""
        weights_dir = self.METHOD_DIR / "weights"
        weights_path = weights_dir / "latest.pth"

        if weights_path.exists():
            return True, ""

        weights_dir.mkdir(parents=True, exist_ok=True)

        try:
            import urllib.request
            print(f"Downloading SLAVC weights from {self.WEIGHTS_URL}...")
            urllib.request.urlretrieve(self.WEIGHTS_URL, str(weights_path))

            if weights_path.exists():
                return True, ""
            else:
                return False, "Download completed but weights not found"

        except Exception as e:
            return False, f"Failed to download weights: {str(e)}"

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run SLAVC sound source localization."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Requires 2 inputs: image file (.png/.jpg) and audio file (.wav)"
            )

        image_path = inputs[0]
        audio_path = inputs[1]

        # Verify inputs
        if not audio_path.suffix.lower() in ['.wav', '.mp3', '.ogg']:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Second input must be an audio file (got {audio_path.suffix})"
            )

        # Ensure weights are available
        weights_ok, weights_error = self._ensure_weights()
        if not weights_ok:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Weights not available: {weights_error}"
            )

        # Get parameters
        alpha = params.get("alpha", 0.4)

        # Setup working directory structure
        work_dir = output_dir / "workdir"
        exec_dir = work_dir / "exec"
        inputs_dir = exec_dir / "inputs"
        frames_dir = inputs_dir / "frames"
        audio_dir = inputs_dir / "audio"
        checkpoints_dir = exec_dir / "checkpoints"
        bin_dir = work_dir / "bin"
        bin_weights_dir = bin_dir / "weights"

        # Create directories
        for d in [frames_dir, audio_dir, checkpoints_dir, bin_weights_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Copy inputs
        shutil.copy(image_path, frames_dir / "input_0.jpg")
        shutil.copy(audio_path, audio_dir / "input_0.wav")

        # Link weights
        weights_src = self.METHOD_DIR / "weights" / "latest.pth"
        weights_dst = bin_weights_dir / "latest.pth"
        if weights_src.exists() and not weights_dst.exists():
            shutil.copy(weights_src, weights_dst)

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "test.py"),
            "--alpha", str(alpha),
            "--test_data_path", str(inputs_dir),
            "--model_dir", str(checkpoints_dir),
            "--experiment_name", "demo",
            "--testset", "demo",
            "--save_visualizations",
            "--relative_prediction"
        ]

        try:
            # Set environment to use correct paths
            env = {
                "PYTHONPATH": str(self.METHOD_DIR),
                "CUDA_VISIBLE_DEVICES": "0"
            }

            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600,
                env={**subprocess.os.environ, **env}
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"SLAVC failed: {result.stderr}"
                )

            # Collect outputs
            outputs = {}
            primary = None

            # Look for output visualizations
            viz_dir = checkpoints_dir / "demo"
            for viz_type in ["av", "obj", "av_obj"]:
                type_dir = viz_dir / viz_type / "viz"
                if type_dir.exists():
                    for f in type_dir.glob("*.jpg"):
                        dst = output_dir / f"{viz_type}_{f.name}"
                        shutil.copy(f, dst)
                        outputs[f"{viz_type}_{f.stem}"] = dst
                        if primary is None and "pred" in f.name:
                            primary = dst

            # Copy SLAVC, OGL, VSL outputs if they exist
            for name in ["slavc", "ogl", "vsl"]:
                src = exec_dir / f"{name}.png"
                if src.exists():
                    dst = output_dir / f"{name}.png"
                    shutil.copy(src, dst)
                    outputs[name] = dst
                    if name == "slavc":
                        primary = dst

            if not primary:
                # Use first available output
                for key, path in outputs.items():
                    if path.exists():
                        primary = path
                        break

            if not primary:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stdout: {result.stdout}, stderr: {result.stderr}"
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
                error_message="SLAVC timed out after 600s"
            )
        except Exception as e:
            import traceback
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"SLAVC error: {str(e)}\n{traceback.format_exc()}"
            )
