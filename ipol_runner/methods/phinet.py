"""Phi-Net InSAR phase denoising adapter (IPOL 2024 article 549).

PyTorch implementation - no TensorFlow dependency.

Based on: F. Sica, G. Gobbi, P. Rizzoli and L. Bruzzone,
"Phi-Net: Deep Residual Learning for InSAR Parameters Estimation,"
IEEE Transactions on Geoscience and Remote Sensing, 2020.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

from ..base import IPOLMethod, MethodResult, MethodCategory, InputType
from ..registry import register


@register
class PhiNetMethod(IPOLMethod):
    """Deep residual learning for InSAR parameters estimation (PyTorch)."""

    METHOD_DIR = Path(__file__).parent.parent.parent / "methods" / "ipol_2024_549_phinet"

    @property
    def name(self) -> str:
        return "phinet"

    @property
    def display_name(self) -> str:
        return "Phi-Net InSAR Phase Denoising"

    @property
    def description(self) -> str:
        return "Deep residual network for estimating interferometric phase and coherence (PyTorch)"

    @property
    def category(self) -> MethodCategory:
        return MethodCategory.PHASE_PROCESSING

    @property
    def input_type(self) -> InputType:
        return InputType.IMAGE_PAIR

    @property
    def input_count(self) -> int:
        return 2  # primary and secondary InSAR images

    @property
    def requirements_file(self):
        # PyTorch requirements instead of TensorFlow
        return None  # Will use torch from main requirements

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {}  # No user-configurable parameters

    def _load_complex_image(self, path: Path) -> np.ndarray:
        """Load a complex-valued image from file."""
        ext = path.suffix.lower()

        if ext == '.npy':
            return np.load(path)
        elif ext in ['.tif', '.tiff']:
            import imageio
            return imageio.imread(path)
        else:
            raise ValueError(f"Unsupported format: {ext}. Use .npy, .tif, or .tiff")

    def _save_visualization(self, output_dir: Path, name: str, data: np.ndarray,
                           vmin: float = None, vmax: float = None, cmap: str = 'viridis'):
        """Save array as PNG visualization."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(name)
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close()

    def run(
        self,
        inputs: List[Path],
        output_dir: Path,
        params: Dict[str, Any]
    ) -> MethodResult:
        """Run Phi-Net InSAR phase denoising using PyTorch."""
        if len(inputs) < 2:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message="Phi-Net requires two input images (primary and secondary)"
            )

        primary_path = inputs[0]
        secondary_path = inputs[1]

        try:
            # Import PyTorch implementation
            from .phinet_pytorch import PhiNetInference, ensure_pytorch_weights

            # Ensure weights are converted
            weights_path = ensure_pytorch_weights(self.METHOD_DIR)

            # Load input images
            logging.info(f"Loading primary image: {primary_path}")
            primary = self._load_complex_image(primary_path)

            logging.info(f"Loading secondary image: {secondary_path}")
            secondary = self._load_complex_image(secondary_path)

            # Validate inputs
            if primary.shape != secondary.shape:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Image shapes must match: {primary.shape} vs {secondary.shape}"
                )

            if not np.iscomplexobj(primary) or not np.iscomplexobj(secondary):
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message="Input images must be complex-valued (InSAR SLC data)"
                )

            min_size = 64
            h, w = primary.shape
            if h < min_size or w < min_size:
                return MethodResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Image size must be at least {min_size}x{min_size}"
                )

            # Run PhiNet inference
            logging.info("Running Phi-Net inference...")
            model = PhiNetInference(weights_path)
            phase, coherence = model.inference(primary, secondary)

            # Create complex denoised interferogram
            denoised = coherence * np.exp(1j * phase).astype(np.complex64)

            # Also compute boxcar for comparison
            from scipy import ndimage as ndi
            win = 5
            intf = primary * np.conj(secondary)
            intf_box = ndi.uniform_filter(np.real(intf), size=win) + \
                       1j * ndi.uniform_filter(np.imag(intf), size=win)
            intensity_primary = ndi.uniform_filter(np.abs(primary) ** 2, size=win)
            intensity_secondary = ndi.uniform_filter(np.abs(secondary) ** 2, size=win)
            boxcar = intf_box / (np.sqrt(intensity_primary * intensity_secondary) + 1e-16)

            # Save outputs
            outputs = {}

            # Save raw data
            np.save(output_dir / "phinet_denoised.npy", denoised)
            outputs["phinet_denoised_npy"] = output_dir / "phinet_denoised.npy"

            np.save(output_dir / "phinet_phase.npy", phase)
            outputs["phinet_phase_npy"] = output_dir / "phinet_phase.npy"

            np.save(output_dir / "phinet_coherence.npy", coherence)
            outputs["phinet_coherence_npy"] = output_dir / "phinet_coherence.npy"

            # Save visualizations
            self._save_visualization(output_dir, "phinet_phase", phase,
                                    vmin=-np.pi, vmax=np.pi, cmap='hsv')
            outputs["phinet_phase"] = output_dir / "phinet_phase.png"

            self._save_visualization(output_dir, "phinet_coherence", coherence,
                                    vmin=0, vmax=1, cmap='gray')
            outputs["phinet_coherence"] = output_dir / "phinet_coherence.png"

            # Boxcar comparison
            self._save_visualization(output_dir, "boxcar_phase", np.angle(boxcar),
                                    vmin=-np.pi, vmax=np.pi, cmap='hsv')
            outputs["boxcar_phase"] = output_dir / "boxcar_phase.png"

            self._save_visualization(output_dir, "boxcar_coherence", np.abs(boxcar),
                                    vmin=0, vmax=1, cmap='gray')
            outputs["boxcar_coherence"] = output_dir / "boxcar_coherence.png"

            # Noisy input
            noisy_phase = np.angle(intf)
            self._save_visualization(output_dir, "noisy_phase", noisy_phase,
                                    vmin=-np.pi, vmax=np.pi, cmap='hsv')
            outputs["noisy_phase"] = output_dir / "noisy_phase.png"

            primary_output = output_dir / "phinet_phase.png"

            logging.info(f"Phi-Net completed successfully. Outputs in {output_dir}")

            return MethodResult(
                success=True,
                output_dir=output_dir,
                primary_output=primary_output,
                outputs=outputs
            )

        except ImportError as e:
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Missing dependency: {e}. Install with: pip install torch h5py matplotlib scipy imageio"
            )
        except Exception as e:
            import traceback
            return MethodResult(
                success=False,
                output_dir=output_dir,
                error_message=f"Phi-Net failed: {str(e)}\n{traceback.format_exc()}"
            )
