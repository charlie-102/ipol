"""Mesh Compression adapter (IPOL 2023 article 418).

Simple quantization-based mesh compression using only numpy.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class MeshCompressionMethod(IPOLMethod):
    """Simple mesh compression with quantization."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2023_418_mesh_compression"

    @property
    def name(self) -> str:
        return "mesh_compression"

    @property
    def display_name(self) -> str:
        return "Mesh Compression"

    @property
    def description(self) -> str:
        return "Compress/decompress triangle meshes (OBJ/OFF)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.RECONSTRUCTION_3D

    @property
    def input_type(self) -> InputType:
        return InputType.DATASET_ID

    @property
    def input_count(self) -> int:
        return 1

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "mode": {
                "type": "choice",
                "choices": ["compress", "decompress"],
                "default": "compress",
                "description": "Compression or decompression"
            },
            "quantization_bits": {
                "type": "int",
                "default": 14,
                "description": "Quantization bits (1-30)"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run mesh compression/decompression."""
        script = self.METHOD_DIR / "mesh_compress.py"
        mode = params.get("mode", "compress")

        input_path = inputs[0] if inputs else None
        if not input_path:
            input_path = self.METHOD_DIR / "codeDemoCPMTools" / "test1.obj"

        if mode == "compress":
            output_file = output_dir / f"{input_path.stem}.mesh"
            cmd = [
                sys.executable, str(script),
                "compress", str(input_path), str(output_file),
                "-q", str(params.get("quantization_bits", 14))
            ]
        else:
            output_file = output_dir / f"{input_path.stem}_decompressed.obj"
            cmd = [
                sys.executable, str(script),
                "decompress", str(input_path), str(output_file)
            ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if output_file.exists():
                return MethodResult(
                    success=True,
                    output_dir=output_dir,
                    primary_output=output_file,
                    outputs={mode: output_file}
                )

            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Failed: {result.stderr or result.stdout}"
            )
        except Exception as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=str(e)
            )
