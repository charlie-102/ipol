"""Survival Forest adapter (IPOL 2024 article 466).

LTRC (Left-Truncated Right-Censored) Survival Forest for survival analysis.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SurvivalForestMethod(IPOLMethod):
    """LTRC Survival Forest for survival analysis.

    Random forest algorithm for left-truncated right-censored survival data.
    Takes tabular data (CSV) with survival information and produces
    survival curves and analysis.
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_466_survival_forest"

    @property
    def name(self) -> str:
        return "survival_forest"

    @property
    def display_name(self) -> str:
        return "LTRC Survival Forest"

    @property
    def description(self) -> str:
        return "Random forest for left-truncated right-censored survival analysis"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.MEDICAL

    @property
    def input_type(self) -> InputType:
        return InputType.SENSOR_DATA  # Tabular/CSV data

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "dataset": {
                "type": "choice",
                "choices": [
                    "larynx",
                    "breast_cancer",
                    "lung_cancer",
                    "convicts",
                    "dictatorship",
                    "flc_chain",
                    "synthetic",
                    "custom"
                ],
                "default": "larynx",
                "description": "Built-in dataset or 'custom' for user-provided CSV"
            },
            "n_estimators": {
                "type": "int",
                "default": 30,
                "min": 1,
                "max": 500,
                "description": "Number of trees in the forest"
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 50,
                "description": "Minimum samples required at leaf node"
            },
            "max_samples": {
                "type": "float",
                "default": 0.89,
                "min": 0.1,
                "max": 1.0,
                "description": "Fraction of samples for bootstrap"
            },
            "entry_col": {
                "type": "str",
                "default": "entry_date",
                "description": "Column name for entry date (left truncation)"
            },
            "duration_col": {
                "type": "str",
                "default": "time",
                "description": "Column name for event/censoring time"
            },
            "event_col": {
                "type": "str",
                "default": "death",
                "description": "Column name for event indicator (1=event, 0=censored)"
            }
        }

    def _get_builtin_dataset(self, name: str) -> Path:
        """Get path to built-in dataset."""
        dataset_map = {
            "larynx": "Larynx Cancer.txt",
            "breast_cancer": "Breast Cancer.txt",
            "lung_cancer": "Lung Cancer.txt",
            "convicts": "Convicts.txt",
            "dictatorship": "dictaorship and democracy.txt",
            "flc_chain": "FLC chain.txt",
            "synthetic": "Synthetic data.txt"
        }
        return self.METHOD_DIR / "data" / dataset_map.get(name, "Larynx Cancer.txt")

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run survival forest analysis."""
        try:
            import numpy as np
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Determine data source
            dataset = params.get("dataset", "larynx")

            if dataset == "custom" and inputs:
                data_path = inputs[0]
            else:
                data_path = self._get_builtin_dataset(dataset)

            if not data_path.exists():
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Dataset not found: {data_path}"
                )

            # Load data
            data = pd.read_csv(data_path, sep=None, engine='python')

            # Get column names
            entry_col = params.get("entry_col", "entry_date")
            duration_col = params.get("duration_col", "time")
            event_col = params.get("event_col", "death")

            # Validate columns exist
            required_cols = [entry_col, duration_col, event_col]
            missing_cols = [c for c in required_cols if c not in data.columns]
            if missing_cols:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Missing columns: {missing_cols}. Available: {list(data.columns)}"
                )

            # Try to import survival_trees, fall back to lifelines if not available
            try:
                from survival_trees import RandomForestLTRCFitter
                from survival_trees.metric import time_dependent_auc
                use_survival_trees = True
            except ImportError:
                use_survival_trees = False

            if use_survival_trees:
                from sklearn.model_selection import train_test_split

                # Prepare data
                y = data[[entry_col, duration_col, event_col]]
                X = data.drop(columns=y.columns.tolist())

                # Split data
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, train_size=0.7, random_state=42
                )

                # Fit model
                model = RandomForestLTRCFitter(
                    n_estimators=params.get("n_estimators", 30),
                    min_impurity_decrease=0.0000001,
                    min_samples_leaf=params.get("min_samples_leaf", 3),
                    max_samples=params.get("max_samples", 0.89)
                )

                model.fit(
                    data.loc[x_train.index],
                    entry_col=entry_col,
                    duration_col=duration_col,
                    event_col=event_col
                )

                # Predict
                survival_function = -np.log(
                    model.predict_cumulative_hazard(x_test).astype(float) + 1e-10
                ).T

                # Calculate AUC
                auc_cd = time_dependent_auc(
                    -survival_function,
                    event_observed=y_test.loc[survival_function.index].iloc[:, 2],
                    censoring_time=y_test.loc[survival_function.index].iloc[:, 1]
                )

                # Plot survival curves
                plt.figure(figsize=(10, 6))
                for i, idx in enumerate(survival_function.index[:5]):
                    plt.plot(survival_function.columns, survival_function.loc[idx], label=f'Subject {i+1}')
                plt.xlabel('Time')
                plt.ylabel('Survival Probability')
                plt.title(f'Survival Curves (AUC: {auc_cd:.3f})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                survival_plot = output_dir / "survival_curves.png"
                plt.savefig(survival_plot, dpi=150)
                plt.close()

            else:
                # Fallback: use lifelines KaplanMeierFitter
                from lifelines import KaplanMeierFitter

                kmf = KaplanMeierFitter()
                kmf.fit(
                    data[duration_col],
                    event_observed=data[event_col],
                    entry=data[entry_col] if entry_col in data.columns else None
                )

                # Plot
                plt.figure(figsize=(10, 6))
                kmf.plot_survival_function()
                plt.xlabel('Time')
                plt.ylabel('Survival Probability')
                plt.title('Kaplan-Meier Survival Curve')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                survival_plot = output_dir / "survival_curves.png"
                plt.savefig(survival_plot, dpi=150)
                plt.close()

                auc_cd = None

            # Create summary statistics
            summary = {
                "dataset": dataset,
                "n_samples": len(data),
                "n_events": int(data[event_col].sum()),
                "censored": int(len(data) - data[event_col].sum()),
                "censoring_rate": float(1 - data[event_col].mean()),
            }
            if auc_cd is not None:
                summary["auc"] = float(auc_cd)

            # Save summary
            summary_file = output_dir / "summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Survival Analysis Summary\n")
                f.write("=" * 40 + "\n\n")
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")

            outputs = {
                "survival_curves": survival_plot,
                "summary": summary_file
            }

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=survival_plot,
                outputs=outputs
            )

        except ImportError as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Missing dependency: {e}. Install with: pip install lifelines pandas matplotlib"
            )
        except Exception as e:
            import traceback
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Survival analysis failed: {str(e)}\n{traceback.format_exc()}"
            )
