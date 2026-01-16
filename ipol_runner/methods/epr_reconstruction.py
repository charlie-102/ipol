"""EPR Image Reconstruction with Total Variation adapter (IPOL 2023 article 414).

TV-EPR: Electron Paramagnetic Resonance image reconstruction.
Converted from MATLAB to Python using finufft for non-uniform FFT.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class EPRReconstructionMethod(IPOLMethod):
    """EPR image reconstruction with Total Variation regularization."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_414_demo-tvepr-master"

    @property
    def name(self) -> str:
        return "epr_reconstruction"

    @property
    def display_name(self) -> str:
        return "TV-EPR Image Reconstruction"

    @property
    def description(self) -> str:
        return "Electron Paramagnetic Resonance reconstruction with Total Variation"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.MEDICAL

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID  # Uses specialized EPR data files

    @property
    def input_count(self) -> int:
        return 0  # Uses built-in datasets, no file input required

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dataset": {
                "type": "choice",
                "choices": [
                    "cnrs_logo_DPPH_2D",
                    "bacteria_inkjet_2D",
                    "cnrs_inkjet_2D",
                    "spades_diamonds_clubs_inkjet_2D",
                    "thin_tube_filled_with_TAM_and_plastic_beads_2D",
                    "tube_filled_with_TAM_and_plastic_beads_2D",
                    "irradiated_phalanx_2D"
                ],
                "default": "cnrs_logo_DPPH_2D",
                "description": "Built-in EPR dataset to use"
            },
            "size": {
                "type": "int",
                "default": 64,
                "description": "Output image size (MxM, must be even)"
            },
            "lambda_tv": {
                "type": "float",
                "default": 1.0,
                "description": "TV regularization weight"
            },
            "niter": {
                "type": "int",
                "default": 2000,
                "description": "Number of iterations"
            },
            "positivity": {
                "type": "bool",
                "default": True,
                "description": "Enable positivity constraint"
            }
        }

    def _find_dataset_files(self, dataset_name: str) -> tuple:
        """Find sinogram and spectrum files for a dataset."""
        data_dir = self.METHOD_DIR / "data" / dataset_name

        if not data_dir.exists():
            return None, None, None

        # Find DTA files
        sinogram_file = None
        spectrum_file = None
        angles_file = None

        for f in data_dir.glob("*.DTA"):
            fname = f.name.lower()
            if "sinogram" in fname:
                sinogram_file = f
            elif "spectrum" in fname:
                spectrum_file = f

        # Find angles file
        for f in data_dir.glob("*.YGF"):
            angles_file = f
            break

        return sinogram_file, spectrum_file, angles_file

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run EPR reconstruction."""
        script = self.METHOD_DIR / "tvepr.py"

        if not script.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Script not found: {script}"
            )

        dataset = params.get("dataset", "cnrs_logo_DPPH_2D")
        sinogram_file, spectrum_file, angles_file = self._find_dataset_files(dataset)

        if not sinogram_file or not spectrum_file:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Dataset files not found for: {dataset}"
            )

        output_file = output_dir / f"epr_{dataset}.png"

        cmd = [
            sys.executable,
            str(script),
            "--sinogram", str(sinogram_file),
            "--spectrum", str(spectrum_file),
            "--output", str(output_file),
            "--size", str(params.get("size", 64)),
            "--lambda-tv", str(params.get("lambda_tv", 1.0)),
            "--niter", str(params.get("niter", 2000)),
        ]

        if angles_file:
            cmd.extend(["--angles", str(angles_file)])

        if params.get("positivity", True):
            cmd.append("--positivity")

        cmd.append("--verbose")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600  # EPR can take a while
            )

            if output_file.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_file,
                    outputs={"reconstruction": output_file}
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"EPR reconstruction failed: {error_msg[:500]}"
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
