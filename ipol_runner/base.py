"""Base classes for IPOL methods."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class MethodCategory(Enum):
    """Categories for IPOL methods based on task type."""
    DENOISING = "denoising"
    CHANGE_DETECTION = "change_detection"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    GENERATION = "generation"
    RECONSTRUCTION_3D = "3d_reconstruction"
    PHASE_PROCESSING = "phase_processing"
    MEDICAL = "medical"


class InputType(Enum):
    """Input types for methods."""
    IMAGE = "image"
    IMAGE_PAIR = "image_pair"
    VIDEO = "video"
    POSE_DATA = "pose_data"
    SENSOR_DATA = "sensor_data"
    DATASET_ID = "dataset_id"


@dataclass
class MethodResult:
    """Result container returned by methods."""
    success: bool
    output_dir: Path
    primary_output: Optional[Path] = None
    outputs: Dict[str, Path] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None


class IPOLMethod(ABC):
    """Abstract base class for all IPOL methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique method identifier (e.g., 'qmsanet')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable method name."""
        pass

    @property
    def description(self) -> str:
        """Short description of what the method does."""
        return ""

    @property
    def category(self) -> MethodCategory:
        """Category/task type of this method."""
        return MethodCategory.DETECTION

    @property
    def input_type(self) -> InputType:
        """Type of input this method expects."""
        return InputType.IMAGE

    @property
    def input_count(self) -> int:
        """Number of input images required (default: 1)."""
        return 1

    @property
    def requirements_file(self) -> Optional[Path]:
        """Path to requirements.txt for this method."""
        return None

    @property
    def requires_cuda(self) -> bool:
        """Whether this method requires NVIDIA CUDA GPU."""
        return False

    @property
    def requires_docker(self) -> bool:
        """Whether this method requires IPOL Docker infrastructure."""
        return False

    @property
    def supports_mps(self) -> bool:
        """Whether this method supports Apple MPS backend.

        Override in subclasses that have been adapted for MPS.
        """
        return False

    @property
    def supports_cpu(self) -> bool:
        """Whether this method can run on CPU (may be slow).

        Most methods support CPU as fallback. Override if not.
        """
        return True

    @property
    def device_choices(self) -> List[str]:
        """Available device choices for this method.

        Returns list of device names this method can use.
        """
        choices = []
        if self.supports_cpu:
            choices.append("cpu")
        if self.supports_mps:
            choices.append("mps")
        if self.requires_cuda:
            choices.append("cuda")
        return choices if choices else ["cpu"]

    @abstractmethod
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema with defaults and descriptions.

        Returns:
            Dict mapping parameter names to their specs:
            {
                "param_name": {
                    "type": "float" | "int" | "str" | "bool" | "choice",
                    "default": <value>,
                    "required": bool,
                    "description": str,
                    "choices": [...],  # if type is "choice"
                    "min": ..., "max": ...,  # if numeric
                }
            }
        """
        pass

    @abstractmethod
    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Execute the method.

        Args:
            inputs: List of input file paths
            output_dir: Directory to write outputs
            params: Method parameters

        Returns:
            MethodResult with outputs and status
        """
        pass

    def validate_inputs(self, inputs: List[Path]) -> Optional[str]:
        """Validate inputs. Returns error message if invalid, None if OK."""
        if len(inputs) != self.input_count:
            return f"Expected {self.input_count} input(s), got {len(inputs)}"
        for inp in inputs:
            if not inp.exists():
                return f"Input file not found: {inp}"
        return None

    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters. Returns error message if invalid, None if OK."""
        schema = self.get_parameters()
        for name, spec in schema.items():
            if spec.get("required", False) and name not in params:
                return f"Required parameter missing: {name}"
        return None
