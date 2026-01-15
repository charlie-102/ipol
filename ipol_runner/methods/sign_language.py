"""Sign Language Segmentation adapters (IPOL 560)."""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SignLanguageLMSLSMethod(IPOLMethod):
    """LMSLS - Linguistically Motivated Sign Language Segmentation."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_560a_sign_lmsls"

    @property
    def name(self) -> str:
        return "sign_lmsls"

    @property
    def display_name(self) -> str:
        return "Sign Language Segmentation (LMSLS)"

    @property
    def description(self) -> str:
        return "Segment sign language video using pose estimation and BIO tagging"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.POSE_DATA

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    @property
    def requires_docker(self) -> bool:
        return True  # Model paths hardcoded for IPOL Docker environment

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "model": {
                "type": "choice",
                "choices": ["model.pth", "model_E1s-1.pth", "model_E4s-1.pth"],
                "default": "model_E1s-1.pth",
                "description": "Pre-trained model to use"
            },
            "sign_b_threshold": {
                "type": "int",
                "default": 60,
                "description": "Probability threshold for sign beginning"
            },
            "sign_o_threshold": {
                "type": "int",
                "default": 50,
                "description": "Probability threshold for sign end"
            },
            "phra_b_threshold": {
                "type": "int",
                "default": 90,
                "description": "Probability threshold for phrase beginning"
            },
            "phra_o_threshold": {
                "type": "int",
                "default": 90,
                "description": "Probability threshold for phrase end"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run LMSLS segmentation. Input should be a pose file."""
        pose_file = inputs[0]

        elan_out = output_dir / "output.eaf"
        sign_srt = output_dir / "signs.srt"
        phra_srt = output_dir / "phrases.srt"

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "sign_language_segmentation" / "main.py"),
            "--pose", str(pose_file),
            "--elan", str(elan_out),
            "--sign_srt", str(sign_srt),
            "--phra_srt", str(phra_srt),
            "--model", params.get("model", "model_E1s-1.pth"),
            "--sign_b_threshold", str(params.get("sign_b_threshold", 60)),
            "--sign_o_threshold", str(params.get("sign_o_threshold", 50)),
            "--phra_b_threshold", str(params.get("phra_b_threshold", 90)),
            "--phra_o_threshold", str(params.get("phra_o_threshold", 90)),
        ]

        try:
            import os
            env = os.environ.copy()
            # Add METHOD_DIR to PYTHONPATH for module imports
            env["PYTHONPATH"] = str(self.METHOD_DIR) + ":" + env.get("PYTHONPATH", "")

            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            outputs = {}
            primary = None

            if elan_out.exists():
                outputs["elan"] = elan_out
                primary = elan_out

            if sign_srt.exists():
                outputs["signs_srt"] = sign_srt

            if phra_srt.exists():
                outputs["phrases_srt"] = phra_srt

            if not outputs:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stderr: {result.stderr}"
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
                error_message="Sign Language LMSLS timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )


@register
class SignLanguageASSLiSUMethod(IPOLMethod):
    """ASSLiSU - Automatic Segmentation of Sign Language into Subtitle-Units."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_560b_sign_asslisu"

    @property
    def name(self) -> str:
        return "sign_asslisu"

    @property
    def display_name(self) -> str:
        return "Sign Language Segmentation (ASSLiSU)"

    @property
    def description(self) -> str:
        return "Segment sign language video into subtitle-units using skeleton data"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.POSE_DATA

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    @property
    def requires_docker(self) -> bool:
        return True  # Model paths hardcoded for IPOL Docker environment

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "which_keypoints": {
                "type": "choice",
                "choices": ["full", "body", "hands", "head", "headbody", "bodyhands"],
                "default": "body",
                "description": "Which keypoints to use"
            },
            "fps": {
                "type": "float",
                "default": 25.0,
                "description": "Frame rate of original video"
            },
            "prob_threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Probability threshold for segmentation"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run ASSLiSU segmentation. Input should be folder with .pkl skeleton files."""
        input_path = inputs[0]

        # If input is a file, use its parent directory
        if input_path.is_file():
            input_folder = input_path.parent
        else:
            input_folder = input_path

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "sign_language_segmentation" / "apply_model.py"),
            "--input_folder", str(input_folder),
            "--output_folder", str(output_dir),
            "--which_keypoints", params.get("which_keypoints", "body"),
            "--fps", str(params.get("fps", 25.0)),
            "--prob_threshold", str(params.get("prob_threshold", 0.5)),
        ]

        try:
            import os
            env = os.environ.copy()
            # Add METHOD_DIR to PYTHONPATH for module imports
            env["PYTHONPATH"] = str(self.METHOD_DIR) + ":" + env.get("PYTHONPATH", "")

            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            outputs = {}
            primary = None

            # Find SRT outputs
            for srt_file in output_dir.glob("*.srt"):
                outputs[srt_file.stem] = srt_file
                if not primary:
                    primary = srt_file

            if not outputs:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"No output generated. stderr: {result.stderr}"
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
                error_message="Sign Language ASSLiSU timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
