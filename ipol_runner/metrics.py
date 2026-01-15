"""Image quality metrics for IPOL methods."""
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array."""
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        return np.array(Image.open(path))


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        raise ValueError(f"Shape mismatch: {img1.shape} vs {img2.shape}")
    return float(np.mean((img1.astype(float) - img2.astype(float)) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio.

    Higher is better. Typical values:
    - < 20 dB: Poor quality
    - 20-30 dB: Acceptable
    - 30-40 dB: Good
    - > 40 dB: Excellent
    """
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / mse_val))


def ssim(img1: np.ndarray, img2: np.ndarray,
         window_size: int = 11,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index.

    Returns value in [-1, 1], where 1 = identical.
    Typical threshold: > 0.95 is very similar.
    """
    try:
        from skimage.metrics import structural_similarity
        # Convert to grayscale if color
        if len(img1.shape) == 3:
            img1_gray = np.mean(img1, axis=2)
            img2_gray = np.mean(img2, axis=2)
        else:
            img1_gray = img1
            img2_gray = img2
        return float(structural_similarity(img1_gray, img2_gray,
                                          win_size=min(window_size, min(img1_gray.shape))))
    except ImportError:
        # Fallback: simplified SSIM
        c1 = (k1 * 255) ** 2
        c2 = (k2 * 255) ** 2

        img1 = img1.astype(float)
        img2 = img2.astype(float)

        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        return float(num / den)


def mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error between two images."""
    return float(np.mean(np.abs(img1.astype(float) - img2.astype(float))))


def compute_all_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute all available metrics between two images."""
    return {
        "MSE": mse(img1, img2),
        "PSNR": psnr(img1, img2),
        "SSIM": ssim(img1, img2),
        "MAE": mae(img1, img2),
    }


def compare_images(path1: Path, path2: Path) -> Dict[str, float]:
    """Load two images and compute all metrics."""
    img1 = load_image(path1)
    img2 = load_image(path2)

    # Resize if needed
    if img1.shape != img2.shape:
        import cv2
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return compute_all_metrics(img1, img2)


def print_metrics(metrics: Dict[str, float], title: str = "Image Quality Metrics"):
    """Pretty print metrics."""
    print(f"\n{title}")
    print("=" * len(title))
    for name, value in metrics.items():
        if name == "PSNR":
            print(f"  {name}: {value:.2f} dB")
        elif name == "SSIM":
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value:.2f}")
