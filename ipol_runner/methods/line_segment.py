"""Line segment detection multi-method adapter (IPOL 2024 article 481)."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class LineSegmentMethod(IPOLMethod):
    """Line segment detection comparison of 8 algorithms."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_481_line_segment"

    # Available algorithms (subfolder names and display names)
    ALGORITHMS = {
        "deeplsd": ("48108-DeepLSD", "DeepLSD (Deep Line Segment Detection)"),
        "edlines": ("48107-EDlines", "EDlines (Edge Drawing Lines)"),
        "afm": ("48106-AFM", "AFM (Attraction Field Map)"),
        "ulsd": ("48105-ULSD", "ULSD (Unified Line Segment Detection)"),
        "sold2": ("48104-SOLD2", "SOLD2 (Self-supervised Line Description)"),
        "mlsd": ("48103-M-LSD", "M-LSD (Mobile Line Segment Detection)"),
        "tplsd": ("48102-TP-LSD", "TP-LSD (Tri-Points Line Segment)"),
        "letr": ("48101-LETR", "LETR (Line Segment Detection Transformer)"),
    }

    @property
    def name(self) -> str:
        return "line_segment"

    @property
    def display_name(self) -> str:
        return "Line Segment Detection (Multi-Method)"

    @property
    def description(self) -> str:
        return "Compare 8 line segment detection algorithms including deep learning methods"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requires_cuda(self) -> bool:
        return True  # Most algorithms use deep learning

    @property
    def requirements_file(self):
        return None  # Each sub-method has its own requirements

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "algorithm": {
                "type": "choice",
                "choices": list(self.ALGORITHMS.keys()),
                "default": "deeplsd",
                "description": "Line segment detection algorithm to use"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run line segment detection."""
        input_path = inputs[0]
        algorithm = params.get("algorithm", "deeplsd")

        if algorithm not in self.ALGORITHMS:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Unknown algorithm: {algorithm}. Choose from: {list(self.ALGORITHMS.keys())}"
            )

        folder_name, display_name = self.ALGORITHMS[algorithm]
        algo_dir = self.METHOD_DIR / folder_name

        if not algo_dir.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Algorithm directory not found: {algo_dir}"
            )

        # Find main.py in the algorithm directory
        main_py = algo_dir / "main.py"
        if not main_py.exists():
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"main.py not found in {algo_dir}"
            )

        # Output files
        out_txt = output_dir / "lines.txt"
        out_svg = output_dir / "lines.svg"
        out_eps = output_dir / "lines.eps"
        out_png = output_dir / "lines.png"

        # Build command - most methods follow similar interface
        cmd = [
            sys.executable,
            str(main_py),
            str(input_path),
            str(out_txt),
            str(out_svg),
            str(out_eps),
            str(out_png)
        ]

        # Special handling for DeepLSD which needs checkpoint path
        if algorithm == "deeplsd":
            ckpt_path = algo_dir / "deeplsd" / "weights" / "deeplsd_md.tar"
            if ckpt_path.exists():
                cmd = [
                    sys.executable,
                    str(main_py),
                    str(input_path),
                    str(ckpt_path),
                    str(out_txt),
                    str(out_svg),
                    str(out_eps),
                    str(out_png)
                ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(algo_dir),
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"{display_name} failed: {result.stderr}"
                )

            # Check for outputs
            outputs = {}
            primary = None

            if out_png.exists():
                outputs["visualization"] = out_png
                primary = out_png

            if out_txt.exists():
                outputs["lines_txt"] = out_txt

            if out_svg.exists():
                outputs["lines_svg"] = out_svg

            if not primary:
                # Check if outputs were created in algo_dir instead
                for out_file in algo_dir.glob("lines.*"):
                    dst = output_dir / out_file.name
                    shutil.copy(out_file, dst)
                    outputs[out_file.stem] = dst
                    if out_file.suffix == ".png" and primary is None:
                        primary = dst

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
                error_message=f"{display_name} timed out after 600s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
