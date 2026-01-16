"""Domain Generalization Segmentation adapter (IPOL 2024 article 499).

Domain generalization for semantic segmentation.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class DomainSegMethod(IPOLMethod):
    """Domain generalization for semantic segmentation.

    Semantic segmentation model trained to generalize across different
    visual domains without domain-specific fine-tuning.
    """

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_499_domain_seg"

    @property
    def name(self) -> str:
        return "domain_seg"

    @property
    def display_name(self) -> str:
        return "Domain Generalization Segmentation"

    @property
    def description(self) -> str:
        return "Semantic segmentation with domain generalization"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.SEGMENTATION

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE

    @property
    def requires_cuda(self) -> bool:
        return True  # Deep learning model

    @property
    def requirements_file(self):
        req_file = self.METHOD_DIR / "requirements.txt"
        return req_file if req_file.exists() else None

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "model": {
                "type": "choice",
                "choices": ["resnet50", "resnet101"],
                "default": "resnet50",
                "description": "Backbone architecture"
            },
            "num_classes": {
                "type": "int",
                "default": 19,
                "min": 2,
                "max": 150,
                "description": "Number of segmentation classes"
            }
        }

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run domain generalization segmentation."""
        if not inputs:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="No input image provided"
            )

        input_path = inputs[0]

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

            h, w = image.shape[:2]
            num_classes = params.get("num_classes", 19)

            # Create placeholder segmentation (actual implementation needs model weights)
            # Using simple color-based clustering as placeholder
            from scipy import ndimage

            # Convert to grayscale and create regions
            gray = np.mean(image, axis=2)

            # Simple thresholding to create regions
            thresholds = np.linspace(0, 255, min(num_classes, 8) + 1)
            segmentation = np.digitize(gray, thresholds[1:-1])

            # Create colormap for visualization
            cmap = plt.cm.get_cmap('tab20', num_classes)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            # Segmentation
            seg_vis = axes[1].imshow(segmentation, cmap=cmap, vmin=0, vmax=num_classes-1)
            axes[1].set_title('Semantic Segmentation')
            axes[1].axis('off')

            # Overlay
            seg_colored = cmap(segmentation / (num_classes - 1))[:, :, :3]
            overlay = image.astype(float) / 255 * 0.5 + seg_colored * 0.5
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            comparison_path = output_dir / "segmentation_result.png"
            plt.savefig(comparison_path, dpi=150)
            plt.close()

            # Save segmentation mask
            mask_path = output_dir / "segmentation_mask.png"
            imageio.imwrite(str(mask_path), segmentation.astype(np.uint8))

            # Save colored segmentation
            colored_path = output_dir / "segmentation_colored.png"
            imageio.imwrite(str(colored_path), (seg_colored * 255).astype(np.uint8))

            outputs = {
                "result": comparison_path,
                "mask": mask_path,
                "colored": colored_path
            }

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=comparison_path,
                outputs=outputs,
                metadata={
                    "model": params.get("model", "resnet50"),
                    "num_classes": num_classes,
                    "note": "Placeholder segmentation - full implementation requires model weights"
                }
            )

        except ImportError as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Missing dependency: {e}. Install with: pip install numpy imageio matplotlib scipy"
            )
        except Exception as e:
            import traceback
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Domain segmentation failed: {str(e)}\n{traceback.format_exc()}"
            )
