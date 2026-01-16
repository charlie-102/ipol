"""Thin-Plate Splines on the Sphere adapter (IPOL preprint 451)."""
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class SphericalSplinesMethod(IPOLMethod):
    """Thin-plate splines interpolation on the sphere."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_pre_451_spherical_splines"

    @property
    def name(self) -> str:
        return "spherical_splines"

    @property
    def display_name(self) -> str:
        return "Thin-Plate Splines on Sphere"

    @property
    def description(self) -> str:
        return "Spherical thin-plate splines for geospatial data interpolation"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID

    @property
    def input_count(self) -> int:
        return 0  # Can generate synthetic data if no input provided

    @property
    def requirements_file(self):
        req_file = self.METHOD_DIR / "requirements.txt"
        if req_file.exists():
            return req_file
        return None

    def _ensure_lookup_tables(self) -> bool:
        """Ensure polylog lookup tables exist."""
        plog2 = self.METHOD_DIR / "plog2.npy"
        plog3 = self.METHOD_DIR / "plog3.npy"
        int_vals = self.METHOD_DIR / "int_vals.npy"

        if plog2.exists() and plog3.exists() and int_vals.exists():
            return True

        # Generate lookup tables if missing
        try:
            subprocess.run(
                [sys.executable, str(self.METHOD_DIR / "GENERATE_PLOG.py")],
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                timeout=300
            )
            subprocess.run(
                [sys.executable, str(self.METHOD_DIR / "GENERATE_INT_VALS.py")],
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                timeout=300
            )
            return plog2.exists() and plog3.exists()
        except Exception:
            return False

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "penalty": {
                "type": "float",
                "default": -3.0,
                "min": -10.0,
                "max": 10.0,
                "description": "Log10 penalty for spherical TPS (lambda = 10^penalty * n)"
            },
            "penalty_cubic": {
                "type": "float",
                "default": -3.0,
                "min": -10.0,
                "max": 10.0,
                "description": "Log10 penalty for natural cubic spline"
            },
            "penalty_tps": {
                "type": "float",
                "default": -3.0,
                "min": -10.0,
                "max": 10.0,
                "description": "Log10 penalty for planar TPS"
            },
            "order": {
                "type": "choice",
                "choices": ["2", "3"],
                "default": "2",
                "description": "Spline order"
            },
            "min_latitude": {
                "type": "float",
                "default": -90.0,
                "min": -90.0,
                "max": 90.0,
                "description": "Minimum latitude for output"
            },
            "max_latitude": {
                "type": "float",
                "default": 90.0,
                "min": -90.0,
                "max": 90.0,
                "description": "Maximum latitude for output"
            },
            "min_longitude": {
                "type": "float",
                "default": -180.0,
                "min": -180.0,
                "max": 180.0,
                "description": "Minimum longitude for output"
            },
            "max_longitude": {
                "type": "float",
                "default": 180.0,
                "min": -180.0,
                "max": 180.0,
                "description": "Maximum longitude for output"
            }
        }

    def _generate_synthetic_data(self, work_dir: Path, n_points: int = 50) -> Path:
        """Generate synthetic sample data for demonstration."""
        import numpy as np
        csv_path = work_dir / "synthetic_data.csv"
        np.random.seed(42)
        lats = np.random.uniform(-90, 90, n_points)
        lons = np.random.uniform(-180, 180, n_points)
        # Generate observations as a function of lat/lon plus noise
        obs = np.sin(np.radians(lats)) * np.cos(np.radians(lons)) + np.random.normal(0, 0.1, n_points)
        with open(csv_path, 'w') as f:
            f.write("latitudes,longitudes,observations\n")
            for lat, lon, o in zip(lats, lons, obs):
                f.write(f"{lat},{lon},{o}\n")
        return csv_path

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run spherical thin-plate splines.

        Input can be a CSV file with columns: latitudes, longitudes, observations
        or a PNG/image file for random sampling on the sphere.
        If no input provided, generates synthetic demo data.
        """
        # Ensure lookup tables exist
        if not self._ensure_lookup_tables():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Failed to generate polylog lookup tables"
            )

        # Copy input to work directory
        work_dir = output_dir / "work"
        work_dir.mkdir(exist_ok=True)

        if inputs:
            input_path = inputs[0]
            if input_path.suffix == '.csv':
                shutil.copy(input_path, work_dir / input_path.name)
            else:
                shutil.copy(input_path, work_dir / f"input{input_path.suffix}")
        else:
            # Generate synthetic data for demo
            self._generate_synthetic_data(work_dir)

        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "tps_min.py"),
            "-p", str(params.get("penalty", -3.0)),
            "-pc", str(params.get("penalty_cubic", -3.0)),
            "-ptps", str(params.get("penalty_tps", -3.0)),
            "-o", str(params.get("order", "2")),
            "-mila", str(params.get("min_latitude", -90.0)),
            "-mala", str(params.get("max_latitude", 90.0)),
            "-milo", str(params.get("min_longitude", -180.0)),
            "-malo", str(params.get("max_longitude", 180.0)),
            "-loc", str(self.METHOD_DIR)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=600
            )

            outputs = {}
            primary = None

            # Collect output images
            output_files = [
                ("data.png", "data"),
                ("interp_sphere.png", "interpolation_spherical"),
                ("interp_plane.png", "interpolation_planar"),
                ("naturalcubicspline.png", "cubic_spline"),
                ("reconstruction_err_sphere.png", "error_spherical"),
                ("reconstruction_err_plane.png", "error_planar"),
                ("diff_sphere_plane.png", "difference"),
            ]

            for src_name, dst_key in output_files:
                src = work_dir / src_name
                if src.exists():
                    dst = output_dir / src_name
                    shutil.copy(src, dst)
                    outputs[dst_key] = dst
                    if primary is None:
                        primary = dst

            if not primary:
                error_msg = result.stderr if result.stderr else result.stdout
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Spherical splines failed: {error_msg[:500]}"
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
                error_message="Spherical splines timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
