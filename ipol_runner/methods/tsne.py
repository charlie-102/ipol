"""t-SNE dimensionality reduction adapter (IPOL 2024 article 528)."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class TSNEMethod(IPOLMethod):
    """t-SNE dimensionality reduction visualization."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_528_tsne"

    @property
    def name(self) -> str:
        return "tsne"

    @property
    def display_name(self) -> str:
        return "t-SNE Dimensionality Reduction"

    @property
    def description(self) -> str:
        return "Compare naive t-SNE vs Barnes-Hut t-SNE for data visualization"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID

    @property
    def requirements_file(self):
        # Note: requires iio package (uses iio_shim)
        return None

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dataset": {
                "type": "choice",
                "choices": ["mnist", "fashion_mnist", "digits"],
                "default": "digits",
                "description": "Dataset to visualize"
            },
            "n_samples": {
                "type": "int",
                "default": 1000,
                "min": 100,
                "max": 10000,
                "description": "Number of samples to use"
            },
            "perplexity": {
                "type": "float",
                "default": 30.0,
                "min": 5.0,
                "max": 100.0,
                "description": "t-SNE perplexity parameter"
            },
            "n_iter": {
                "type": "int",
                "default": 1000,
                "min": 250,
                "max": 5000,
                "description": "Number of iterations"
            },
            "learning_rate": {
                "type": "float",
                "default": 200.0,
                "min": 10.0,
                "max": 1000.0,
                "description": "Learning rate"
            },
            "use_pca": {
                "type": "bool",
                "default": True,
                "description": "Apply PCA preprocessing"
            },
            "pca_components": {
                "type": "int",
                "default": 50,
                "min": 2,
                "max": 100,
                "description": "Number of PCA components"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run t-SNE comparison."""
        # Get parameters
        dataset = params.get("dataset", "digits")
        n_samples = params.get("n_samples", 1000)
        perplexity = params.get("perplexity", 30.0)
        n_iter = params.get("n_iter", 1000)
        learning_rate = params.get("learning_rate", 200.0)
        use_pca = params.get("use_pca", True)
        pca_components = params.get("pca_components", 50)

        # Build command
        output_naive = output_dir / "tsne_naive.png"
        output_bh = output_dir / "tsne_barnes_hut.png"

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "naive_tsne_vs_barnes_hut_tsne.py"),
            "--dataset", dataset,
            "--n_samples", str(n_samples),
            "--perplexity", str(perplexity),
            "--n_iter", str(n_iter),
            "--learning_rate", str(learning_rate),
            "--output_naive", str(output_naive),
            "--output_bh", str(output_bh)
        ]
        if use_pca:
            cmd.extend(["--pca_components", str(pca_components)])

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"t-SNE failed: {result.stderr}"
                )

            # Check for outputs
            outputs = {}
            primary = None

            for out_file in output_dir.glob("*.png"):
                outputs[out_file.stem] = out_file
                if primary is None:
                    primary = out_file

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
                error_message="t-SNE timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
