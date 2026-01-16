"""Noisesniffer image forgery detection adapter (IPOL 2024 article 462)."""
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class NoisesnifferMethod(IPOLMethod):
    """Image forgery detection based on noise inspection."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_462_noisesniffer"

    @property
    def name(self) -> str:
        return "noisesniffer"

    @property
    def display_name(self) -> str:
        return "Noisesniffer Forgery Detection"

    @property
    def description(self) -> str:
        return "Detect image forgeries through noise analysis using local noise anomalies"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "block_size": {
                "type": "int",
                "default": 3,
                "min": 2,
                "max": 10,
                "description": "Block size for noise analysis"
            },
            "blocks_per_bin": {
                "type": "int",
                "default": 20000,
                "min": 1000,
                "max": 100000,
                "description": "Number of blocks per bin"
            },
            "low_freq_percentile": {
                "type": "float",
                "default": 0.1,
                "min": 0.01,
                "max": 1.0,
                "description": "Percentile of blocks with lowest energy in low frequencies"
            },
            "std_percentile": {
                "type": "float",
                "default": 0.5,
                "min": 0.01,
                "max": 1.0,
                "description": "Percentile of blocks with lowest standard deviation"
            },
            "cell_size": {
                "type": "int",
                "default": 100,
                "min": 10,
                "max": 500,
                "description": "Cell size for NFA computation (region growing)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run Noisesniffer forgery detection."""
        input_path = inputs[0]

        # Get parameters
        w = params.get("block_size", 3)
        b = params.get("blocks_per_bin", 20000)
        n = params.get("low_freq_percentile", 0.1)
        m = params.get("std_percentile", 0.5)
        W = params.get("cell_size", 100)

        # Build command
        cmd = [
            sys.executable,
            str(self.METHOD_DIR / "Noisesniffer.py"),
            str(input_path),
            "-w", str(w),
            "-b", str(b),
            "-n", str(n),
            "-m", str(m),
            "-W", str(W)
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.METHOD_DIR),
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Noisesniffer failed: {result.stderr}"
                )

            # Move output files to output_dir
            outputs = {}
            primary = None

            # Output files are created in METHOD_DIR
            distributions = self.METHOD_DIR / "output_distributions.png"
            mask = self.METHOD_DIR / "output_mask.png"
            nfa = self.METHOD_DIR / "NFA.txt"

            if distributions.exists():
                dst = output_dir / "distributions.png"
                shutil.copy(distributions, dst)
                outputs["distributions"] = dst
                primary = dst
                distributions.unlink()

            if mask.exists():
                dst = output_dir / "mask.png"
                shutil.copy(mask, dst)
                outputs["mask"] = dst
                if primary is None:
                    primary = dst
                mask.unlink()

            if nfa.exists():
                dst = output_dir / "NFA.txt"
                shutil.copy(nfa, dst)
                outputs["nfa"] = dst
                nfa.unlink()

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
                error_message="Noisesniffer timed out after 300s"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
