"""Semantic Segmentation Zoo adapter (IPOL 2023 article 447).

Multiple segmentation architectures on ADE20K dataset.
"""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SegmentationZooMethod(IPOLMethod):
    """Semantic segmentation with multiple architectures (FCN, PSPNet, DeepLabV3+, SETR, Segformer)."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_447_segmentation_zoo"

    @property
    def name(self) -> str:
        return "segmentation_zoo"

    @property
    def display_name(self) -> str:
        return "Semantic Segmentation Zoo"

    @property
    def description(self) -> str:
        return "Compare FCN, PSPNet, DeepLabV3+, SETR, Segformer on ADE20K"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        # mmsegmentation has its own complex requirements
        return None

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "model": {
                "type": "choice",
                "choices": ["FCN", "PSPnet", "Deeplabv3plus", "SETR", "Segformer"],
                "default": "FCN",
                "description": "Segmentation architecture"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run semantic segmentation."""
        input_path = inputs[0]
        model = params.get("model", "FCN")

        # Check for checkpoint files
        checkpoints_dir = self.METHOD_DIR / "checkpoints"
        checkpoint_map = {
            "FCN": "fcn_r101-d8_512x512_160k_ade20k_20200615_105816-fd192bd5.pth",
            "PSPnet": "pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth",
            "Deeplabv3plus": "deeplabv3plus_r101-d8_512x512_160k_ade20k_20200615_123232-38ed86bb.pth",
            "SETR": "setr_naive_512x512_160k_b16_ade20k_20210619_191258-061f24f5.pth",
            "Segformer": "segformer_mit-b4_512x512_160k_ade20k_20210728_183055-7f509d7d.pth"
        }

        checkpoint_file = checkpoints_dir / checkpoint_map.get(model, "")
        if not checkpoint_file.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Checkpoint not found: {checkpoint_file.name}. Download from IPOL and place in {checkpoints_dir}"
            )

        # Copy input to method directory as input_0.png (script expects this)
        method_input = self.METHOD_DIR / "input_0.png"
        shutil.copy(str(input_path), str(method_input))

        try:
            cmd = [
                sys.executable,
                str(self.METHOD_DIR / "run.py"),
                "--net", model
            ]

            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Collect outputs from method directory
            outputs = {}
            primary = None

            # Expected outputs
            output_files = {
                "seg": self.METHOD_DIR / "seg.png",
                "entropy": self.METHOD_DIR / "entropy.png",
                "label": self.METHOD_DIR / "label.png",
                "img": self.METHOD_DIR / "img.png"
            }

            for name, src in output_files.items():
                if src.exists():
                    dst = output_dir / f"{name}.png"
                    shutil.copy(str(src), str(dst))
                    outputs[name] = dst
                    if name == "seg":
                        primary = dst

            # Cleanup method directory
            for src in output_files.values():
                if src.exists():
                    src.unlink()
            if method_input.exists():
                method_input.unlink()

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
                error_message=f"Segmentation failed: {error_msg[:500]}"
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
        finally:
            # Ensure cleanup
            if method_input.exists():
                method_input.unlink()
