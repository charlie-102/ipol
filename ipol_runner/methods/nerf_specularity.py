"""NeRF Specularity adapter (IPOL 562)."""
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class NeRFSpecularityMethod(IPOLMethod):
    """NeRF Specularity - Compare Ref-NeRF and NRFF methods."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2025_562_nerf_specularity"
    DATASETS = ["toaster", "ani", "ball"]

    @property
    def name(self) -> str:
        return "nerf_specularity"

    @property
    def display_name(self) -> str:
        return "NeRF Specularity (Ref-NeRF vs NRFF)"

    @property
    def description(self) -> str:
        return "Render specular materials using NeRF with GT/RefNeRF/NRFF comparison"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID

    @property
    def input_count(self) -> int:
        return 0  # Uses predefined datasets, no file input

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dataset": {
                "type": "choice",
                "choices": self.DATASETS,
                "default": "toaster",
                "required": True,
                "description": "Dataset to render (toaster, ani, or ball)"
            },
            "azimuth": {
                "type": "int",
                "default": 0,
                "description": "Azimuth angle for rendering (valid: -160 to 160, step 10)"
            }
        }

    def validate_inputs(self, inputs: List[Path]) -> str:
        # No file inputs required
        return None

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        import os
        import tempfile

        dataset = params.get("dataset", "toaster")
        azimuth = int(params.get("azimuth", 0))  # Must be integer for demo_azimuth lookup

        # The script reads dataset from a file via: dataset=$(cat $input_0)
        # Create a temp file with the dataset name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(dataset)
            input_file = f.name

        # Run the bash script
        cmd = ["bash", str(self.METHOD_DIR / "main.sh"), str(azimuth)]

        try:
            # Set input_0 to the temp file path, and bin to METHOD_DIR
            env = {"input_0": input_file, "bin": str(self.METHOD_DIR)}
            env.update(os.environ)

            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600,
                env=env
            )

            # Move outputs from demo_output to output_dir
            demo_output = self.METHOD_DIR / "demo_output"
            outputs = {}
            primary = None

            if demo_output.exists():
                import shutil
                for item in demo_output.iterdir():
                    if item.is_dir():
                        dst_dir = output_dir / item.name
                        shutil.copytree(item, dst_dir, dirs_exist_ok=True)
                        # Get first image from each directory
                        images = list(dst_dir.glob("*.png"))
                        if images:
                            outputs[item.name] = images[0]
                            if item.name == "gt_rgb" and not primary:
                                primary = images[0]
                    else:
                        dst = output_dir / item.name
                        shutil.copy(item, dst)
                        outputs[item.stem] = dst

            if not primary and outputs:
                primary = list(outputs.values())[0]

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
                error_message="NeRF Specularity timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
