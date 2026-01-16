"""One-Shot Federated Learning adapter (IPOL 2023 article 440).

FESC algorithm for federated learning with heterogeneous data distribution.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class FederatedLearningMethod(IPOLMethod):
    """One-shot federated learning simulation."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_440_federated_learning"

    @property
    def name(self) -> str:
        return "federated_learning"

    @property
    def display_name(self) -> str:
        return "One-Shot Federated Learning (FESC)"

    @property
    def description(self) -> str:
        return "Compare Centralized, Federated, and FESC algorithms for heterogeneous data"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION  # Mathematical simulation

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID  # No image input, uses parameters

    @property
    def input_count(self) -> int:
        return 0  # No file inputs required

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "src" / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "n_nodes": {
                "type": "int",
                "default": 78,
                "description": "Number of federated nodes (2-300)"
            },
            "prob_law": {
                "type": "choice",
                "choices": [
                    "log_normal_low_var", "log_normal_high_var",
                    "gaussian_low_var", "gaussian_med_var", "gaussian_high_var",
                    "poisson", "uniform_low_var", "uniform_high_var",
                    "laplacian_low_var", "laplacian_med_var", "laplacian_high_var",
                    "pareto", "weibull"
                ],
                "default": "log_normal_low_var",
                "description": "Distribution for samples per node"
            },
            "feature_dims": {
                "type": "int",
                "default": 10,
                "description": "Feature space dimension (10-100)"
            },
            "n_montecarlo": {
                "type": "int",
                "default": 10,
                "description": "Monte Carlo iterations"
            },
            "seed": {
                "type": "int",
                "default": 42,
                "description": "Random seed for reproducibility"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run federated learning simulation."""
        script = self.METHOD_DIR / "federated_learning.py"

        if not script.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Script not found: {script}"
            )

        output_file = output_dir / "fesc_comparison.png"

        cmd = [
            sys.executable,
            str(script),
            "--n_nodes", str(params.get("n_nodes", 78)),
            "--prob_law", str(params.get("prob_law", "log_normal_low_var")),
            "--feature_dims", str(params.get("feature_dims", 10)),
            "--n_montecarlo", str(params.get("n_montecarlo", 10)),
            "--seed", str(params.get("seed", 42)),
            "--output", str(output_file)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=120
            )

            if output_file.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_file,
                    outputs={"comparison": output_file}
                )

            error_msg = result.stderr if result.stderr else result.stdout
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Simulation failed: {error_msg[:500]}"
            )

        except subprocess.TimeoutExpired:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Method timed out after 120s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
