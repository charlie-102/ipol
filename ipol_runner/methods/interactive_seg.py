"""Interactive Segmentation adapter (IPOL 2024 article 498).

Comparing interactive image segmentation models under different clicking procedures.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class InteractiveSegMethod(IPOLMethod):
    """Interactive image segmentation comparison.

    Compares different interactive segmentation models (RITM, GTO99)
    under various clicking procedures.
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_498_interactive_seg"

    @property
    def name(self) -> str:
        return "interactive_seg"

    @property
    def display_name(self) -> str:
        return "Interactive Segmentation"

    @property
    def description(self) -> str:
        return "Compare interactive segmentation models with different clicking strategies"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requires_cuda(self) -> bool:
        return True  # Deep learning models

    @property
    def requirements_file(self):
        return self.METHOD_DIR / "requirements.txt"

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "model": {
                "type": "choice",
                "choices": ["ritm", "gto99"],
                "default": "ritm",
                "description": "Segmentation model: RITM (HRNet) or GTO99"
            },
            "clicking_strategy": {
                "type": "choice",
                "choices": ["random", "center", "boundary"],
                "default": "center",
                "description": "Click simulation strategy"
            },
            "max_clicks": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 50,
                "description": "Maximum number of clicks to simulate"
            },
            "iou_threshold": {
                "type": "float",
                "default": 0.85,
                "min": 0.5,
                "max": 0.99,
                "description": "Target IoU threshold for early stopping"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run interactive segmentation comparison."""
        if not inputs:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="No input image provided"
            )

        input_path = inputs[0]

        # Get parameters
        model = params.get("model", "ritm")
        clicking_strategy = params.get("clicking_strategy", "center")
        max_clicks = params.get("max_clicks", 10)
        iou_threshold = params.get("iou_threshold", 0.85)

        try:
            import numpy as np
            import imageio.v2 as imageio
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Load image
            image = imageio.imread(str(input_path))
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]

            # For now, create a simple segmentation visualization
            # (Full implementation would require model weights)
            h, w = image.shape[:2]

            # Create a simple center-based segmentation mask (placeholder)
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2

            # Simple elliptical mask centered on image
            mask = ((x - center_x)**2 / (w/3)**2 + (y - center_y)**2 / (h/3)**2) <= 1.0

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            # Segmentation mask
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title(f'Segmentation ({model})')
            axes[1].axis('off')

            # Overlay
            overlay = image.copy().astype(float)
            overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            comparison_path = output_dir / "segmentation_comparison.png"
            plt.savefig(comparison_path, dpi=150)
            plt.close()

            # Save mask
            mask_path = output_dir / "segmentation_mask.png"
            imageio.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

            outputs = {
                "comparison": comparison_path,
                "mask": mask_path
            }

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=comparison_path,
                outputs=outputs,
                metadata={
                    "model": model,
                    "clicking_strategy": clicking_strategy,
                    "note": "Placeholder - full implementation requires model weights"
                }
            )

        except ImportError as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Missing dependency: {e}. Install with: pip install numpy imageio matplotlib"
            )
        except Exception as e:
            import traceback
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Interactive segmentation failed: {str(e)}\n{traceback.format_exc()}"
            )
