"""Simplified Python implementation of image abstraction.

Note: This is a simplified alternative to the full Tree of Shapes algorithm.
For full functionality, use the C++ backend.

This provides basic abstraction effects using:
- Bilateral filtering for edge-preserving smoothing
- Quantization for color reduction
- Edge enhancement for artistic effects
"""
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import logging


def bilateral_filter(
    image: np.ndarray,
    sigma_spatial: float = 10.0,
    sigma_color: float = 30.0,
    window_size: int = 5
) -> np.ndarray:
    """Apply bilateral filter for edge-preserving smoothing.

    This is a simplified bilateral filter using Gaussian weighting.

    Args:
        image: Input image (H, W, C) or (H, W) in [0, 255]
        sigma_spatial: Spatial Gaussian sigma
        sigma_color: Color/intensity Gaussian sigma
        window_size: Filter window size

    Returns:
        Filtered image
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    h, w, c = image.shape
    half_window = window_size // 2
    output = np.zeros_like(image, dtype=np.float64)

    # Pad image
    padded = np.pad(image, ((half_window, half_window), (half_window, half_window), (0, 0)), mode='reflect')

    # Precompute spatial weights
    y_coords, x_coords = np.mgrid[-half_window:half_window+1, -half_window:half_window+1]
    spatial_weights = np.exp(-(x_coords**2 + y_coords**2) / (2 * sigma_spatial**2))

    for y in range(h):
        for x in range(w):
            # Extract window
            window = padded[y:y+window_size, x:x+window_size, :]
            center = image[y, x, :]

            # Color weights
            color_diff = np.sum((window - center)**2, axis=2)
            color_weights = np.exp(-color_diff / (2 * sigma_color**2))

            # Combined weights
            weights = spatial_weights * color_weights
            weights_sum = np.sum(weights) + 1e-8

            # Apply filter
            for ch in range(c):
                output[y, x, ch] = np.sum(window[:, :, ch] * weights) / weights_sum

    return output.squeeze() if c == 1 else output


def quantize_colors(
    image: np.ndarray,
    n_colors: int = 16
) -> np.ndarray:
    """Reduce number of colors in image.

    Args:
        image: Input image in [0, 255]
        n_colors: Number of colors in output

    Returns:
        Quantized image
    """
    # Simple quantization by reducing bit depth
    levels = 256 // n_colors
    quantized = (image // levels) * levels + levels // 2
    return np.clip(quantized, 0, 255)


def enhance_edges(
    image: np.ndarray,
    strength: float = 0.5
) -> np.ndarray:
    """Enhance edges for artistic effect.

    Args:
        image: Input image in [0, 255]
        strength: Edge enhancement strength (0-1)

    Returns:
        Edge-enhanced image
    """
    if image.ndim == 2:
        gray = image
    else:
        gray = np.mean(image, axis=2)

    # Detect edges using Sobel
    sobel_x = ndimage.sobel(gray, axis=1)
    sobel_y = ndimage.sobel(gray, axis=0)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalize edges
    edges = edges / (edges.max() + 1e-8)

    # Apply edge darkening
    if image.ndim == 2:
        result = image * (1 - strength * edges)
    else:
        edges_3d = edges[:, :, np.newaxis]
        result = image * (1 - strength * edges_3d)

    return np.clip(result, 0, 255)


def simple_segmentation(
    image: np.ndarray,
    min_area: int = 100
) -> np.ndarray:
    """Simple segmentation by region growing.

    Args:
        image: Input image in [0, 255]
        min_area: Minimum region area

    Returns:
        Segmented image with flat colors per region
    """
    if image.ndim == 2:
        gray = image
    else:
        gray = np.mean(image, axis=2)

    # Quantize to reduce regions
    labels, num_labels = ndimage.label(quantize_colors(gray, 8) // 32)

    # Average colors per region
    if image.ndim == 2:
        output = np.zeros_like(image, dtype=np.float64)
        for i in range(1, num_labels + 1):
            mask = labels == i
            if np.sum(mask) >= min_area:
                output[mask] = np.mean(image[mask])
            else:
                output[mask] = image[mask]
    else:
        output = np.zeros_like(image, dtype=np.float64)
        for i in range(1, num_labels + 1):
            mask = labels == i
            if np.sum(mask) >= min_area:
                for c in range(image.shape[2]):
                    output[mask, c] = np.mean(image[mask, c])
            else:
                output[mask] = image[mask]

    return output


def abstraction_effect(
    image: np.ndarray,
    task: str = "abstraction",
    smoothing_strength: float = 0.5,
    edge_strength: float = 0.3,
    n_colors: int = 32
) -> np.ndarray:
    """Apply abstraction effect to image.

    Args:
        image: Input image (H, W, C) in [0, 255]
        task: One of "abstraction", "watercolor", "smoothing"
        smoothing_strength: Amount of smoothing (0-1)
        edge_strength: Amount of edge darkening (0-1)
        n_colors: Number of colors for quantization

    Returns:
        Abstracted image
    """
    image = image.astype(np.float64)

    if task == "abstraction":
        # Bilateral filter + quantization + edges
        sigma_spatial = 5 + smoothing_strength * 20
        sigma_color = 20 + smoothing_strength * 40

        # Simplified bilateral using Gaussian blur
        smoothed = ndimage.gaussian_filter(image, sigma=sigma_spatial)

        # Mix original and smoothed based on color difference
        if image.ndim == 3:
            diff = np.sum(np.abs(image - smoothed), axis=2, keepdims=True)
        else:
            diff = np.abs(image - smoothed)
        mix = np.exp(-diff / (2 * sigma_color))
        result = image * (1 - mix) + smoothed * mix

        # Quantize
        result = quantize_colors(result, n_colors)

        # Enhance edges
        result = enhance_edges(result, edge_strength)

    elif task == "watercolor":
        # More smoothing, less edges
        smoothed = ndimage.median_filter(image, size=5)
        smoothed = ndimage.gaussian_filter(smoothed, sigma=3)

        # Quantize heavily
        result = quantize_colors(smoothed, max(8, n_colors // 2))

        # Light edge enhancement
        result = enhance_edges(result, edge_strength * 0.5)

    elif task == "smoothing":
        # Just bilateral-like smoothing
        sigma = 5 + smoothing_strength * 15
        result = ndimage.gaussian_filter(image, sigma=sigma)

    else:
        logging.warning(f"Unknown task '{task}', using abstraction")
        return abstraction_effect(image, "abstraction", smoothing_strength, edge_strength, n_colors)

    return np.clip(result, 0, 255)


def image_abstraction(
    image: np.ndarray,
    task: str = "abstraction",
    min_area: float = 0.01,
    scale_ratio: int = 3,
    threshold: float = 0.5,
    alpha: float = 1.0
) -> np.ndarray:
    """Main abstraction function.

    Note: This is a simplified version. For full Tree of Shapes
    functionality, use the C++ backend.

    Args:
        image: Input image (H, W, C) in [0, 255] as uint8 or float
        task: "abstraction", "watercolor", "shaking", "smoothing"
        min_area: Minimum area threshold (relative to image size)
        scale_ratio: Scale ratio for multi-scale processing
        threshold: Effect threshold
        alpha: Blending alpha

    Returns:
        Abstracted image in [0, 255]
    """
    image = image.astype(np.float64)
    h, w = image.shape[:2]

    # Map parameters to effect strengths
    smoothing_strength = threshold
    edge_strength = threshold * 0.5
    n_colors = max(8, int(64 * (1 - min_area * 10)))

    # Apply effect
    result = abstraction_effect(
        image,
        task=task,
        smoothing_strength=smoothing_strength,
        edge_strength=edge_strength,
        n_colors=n_colors
    )

    # Blend with original
    if alpha < 1.0:
        result = alpha * result + (1 - alpha) * image

    return np.clip(result, 0, 255)


if __name__ == "__main__":
    import imageio.v2 as imageio
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_abstraction_python.py <input> [output] [task]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "abstraction.png"
    task = sys.argv[3] if len(sys.argv) > 3 else "abstraction"

    logging.basicConfig(level=logging.INFO)

    image = imageio.imread(input_path)
    result = image_abstraction(image, task=task)

    imageio.imwrite(output_path, result.astype(np.uint8))
    print(f"Result saved to {output_path}")
