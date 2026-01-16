"""Python implementation of Dark Channel Prior dehazing.

Based on: K. He, J. Sun, and X. Tang,
"Single Image Haze Removal Using Dark Channel Prior,"
IEEE TPAMI, 2011.

IPOL article: Jose-Luis Lisani, 2024
https://www.ipol.im/pub/art/2024/530/
"""
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional
import logging


def get_dark_channel(image: np.ndarray, patch_radius: int = 7) -> np.ndarray:
    """Compute dark channel of an image.

    Dark channel is the minimum value over local patch and color channels.

    Args:
        image: Input image (H, W, C) or (H, W) in range [0, 255]
        patch_radius: Radius of local patch for minimum filter

    Returns:
        Dark channel image (H, W) in range [0, 255]
    """
    if image.ndim == 2:
        min_rgb = image
    else:
        # Minimum over color channels
        min_rgb = np.min(image, axis=2)

    # Minimum over local patch using minimum filter
    patch_size = 2 * patch_radius + 1
    dark_channel = ndimage.minimum_filter(min_rgb, size=patch_size, mode='reflect')

    return dark_channel


def estimate_ambient_light(
    dark_channel: np.ndarray,
    image: np.ndarray,
    top_percent: float = 0.1,
    exclude_saturated: bool = True
) -> np.ndarray:
    """Estimate atmospheric light from brightest pixels in dark channel.

    Args:
        dark_channel: Dark channel image (H, W)
        image: Original image (H, W, C) in range [0, 255]
        top_percent: Percentage of brightest pixels to use
        exclude_saturated: Whether to exclude saturated (255) pixels

    Returns:
        Ambient light values (C,) or scalar for grayscale
    """
    h, w = dark_channel.shape
    n_pixels = int(h * w * top_percent / 100.0)
    n_pixels = max(1, n_pixels)

    # Get indices of brightest pixels in dark channel
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[::-1]  # Descending order

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    n_channels = image.shape[2]
    ambient = np.zeros(n_channels)
    count = 0

    for idx in indices:
        if count >= n_pixels:
            break

        y, x = idx // w, idx % w
        pixel = image[y, x]

        # Skip saturated pixels if requested
        if exclude_saturated and np.any(pixel >= 255):
            continue

        ambient += pixel
        count += 1

    if count == 0:
        # Fallback: use brightest pixel
        idx = indices[0]
        y, x = idx // w, idx % w
        ambient = image[y, x].astype(float)
    else:
        ambient /= count

    return ambient.squeeze() if n_channels == 1 else ambient


def get_transmission_map(
    image: np.ndarray,
    ambient: np.ndarray,
    patch_radius: int = 7,
    omega: float = 0.95
) -> np.ndarray:
    """Compute transmission map.

    Args:
        image: Input image (H, W, C) in range [0, 255]
        ambient: Ambient light values (C,)
        patch_radius: Radius for minimum filter
        omega: Haze removal amount (0-1), higher = more removal

    Returns:
        Transmission map (H, W) in range [0, 1]
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        ambient = np.array([ambient])

    # Normalize by ambient light
    normalized = image / (ambient + 1e-8)

    # Minimum over channels
    min_normalized = np.min(normalized, axis=2)

    # Minimum over local patch
    patch_size = 2 * patch_radius + 1
    min_patch = ndimage.minimum_filter(min_normalized, size=patch_size, mode='reflect')

    # Transmission map
    transmission = 1.0 - omega * min_patch

    return transmission


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 30,
    eps: float = 0.0001
) -> np.ndarray:
    """Apply guided filter for edge-preserving smoothing.

    Args:
        guide: Guide image (H, W), normalized to [0, 1]
        src: Source image to filter (H, W)
        radius: Filter radius
        eps: Regularization parameter

    Returns:
        Filtered image (H, W)
    """
    # Box filter using uniform filter
    box_size = 2 * radius + 1

    def box_filter(img):
        return ndimage.uniform_filter(img, size=box_size, mode='reflect')

    # Mean of guide
    mean_I = box_filter(guide)

    # Mean of guide squared
    mean_I2 = box_filter(guide * guide)

    # Variance of guide
    var_I = mean_I2 - mean_I * mean_I

    # Mean of source
    mean_p = box_filter(src)

    # Mean of guide * source
    mean_Ip = box_filter(guide * src)

    # Covariance
    cov_Ip = mean_Ip - mean_I * mean_p

    # Linear coefficients
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # Mean of coefficients
    mean_a = box_filter(a)
    mean_b = box_filter(b)

    # Output
    output = mean_a * guide + mean_b

    return output


def refine_transmission(
    transmission: np.ndarray,
    image: np.ndarray,
    radius: int = 30,
    eps: float = 0.0001
) -> np.ndarray:
    """Refine transmission map using guided filter.

    Args:
        transmission: Initial transmission map (H, W)
        image: Original image for guidance (H, W, C) in [0, 255]
        radius: Guided filter radius
        eps: Regularization parameter

    Returns:
        Refined transmission map (H, W)
    """
    # Use intensity as guide
    if image.ndim == 3:
        guide = np.mean(image, axis=2) / 255.0
    else:
        guide = image / 255.0

    refined = guided_filter(guide, transmission, radius, eps)

    return refined


def recover_radiance(
    image: np.ndarray,
    transmission: np.ndarray,
    ambient: np.ndarray,
    t0: float = 0.1
) -> np.ndarray:
    """Recover scene radiance (dehazed image).

    Args:
        image: Input hazy image (H, W, C) in [0, 255]
        transmission: Transmission map (H, W) in [0, 1]
        ambient: Ambient light (C,) in [0, 255]
        t0: Minimum transmission to avoid division by zero

    Returns:
        Dehazed image (H, W, C) in [0, 255]
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        ambient = np.array([ambient])

    # Normalize
    image_norm = image / 255.0
    ambient_norm = ambient / 255.0

    # Clip transmission
    t = np.maximum(transmission, t0)[:, :, np.newaxis]

    # Recover radiance
    radiance = (image_norm - ambient_norm) / t + ambient_norm

    # Clip to valid range
    radiance = np.clip(radiance * 255.0, 0, 255)

    return radiance.squeeze() if radiance.shape[2] == 1 else radiance


def dehaze_dark_channel_prior(
    image: np.ndarray,
    omega: float = 0.95,
    patch_radius: int = 7,
    guided_radius: int = 30,
    guided_eps: float = 0.0001,
    t0: float = 0.1,
    exclude_saturated: bool = True,
    top_percent: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dehaze image using Dark Channel Prior.

    Args:
        image: Input hazy image (H, W, C) or (H, W), uint8 or float [0-255]
        omega: Haze removal amount (0=none, 1=full)
        patch_radius: Radius for dark channel computation
        guided_radius: Radius for guided filter refinement
        guided_eps: Regularization for guided filter
        t0: Minimum transmission value
        exclude_saturated: Exclude saturated pixels in ambient estimation
        top_percent: Percentage of brightest dark channel pixels for ambient

    Returns:
        dehazed: Dehazed image (H, W, C) in [0, 255]
        transmission: Refined transmission map (H, W) in [0, 1]
        ambient: Estimated ambient light (C,)
    """
    # Ensure float
    image = image.astype(np.float64)

    logging.info("Computing dark channel...")
    dark_channel = get_dark_channel(image, patch_radius)

    logging.info("Estimating ambient light...")
    ambient = estimate_ambient_light(
        dark_channel, image, top_percent, exclude_saturated
    )
    logging.info(f"Ambient light: {ambient}")

    logging.info("Computing transmission map...")
    transmission = get_transmission_map(image, ambient, patch_radius, omega)

    logging.info("Refining transmission with guided filter...")
    transmission_refined = refine_transmission(
        transmission, image, guided_radius, guided_eps
    )

    logging.info("Recovering scene radiance...")
    dehazed = recover_radiance(image, transmission_refined, ambient, t0)

    return dehazed, transmission_refined, ambient


if __name__ == "__main__":
    # Test with a sample image
    import imageio.v2 as imageio
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dark_channel_python.py <input_image> [output_image]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dehazed.png"

    logging.basicConfig(level=logging.INFO)

    image = imageio.imread(input_path)
    dehazed, transmission, ambient = dehaze_dark_channel_prior(image)

    imageio.imwrite(output_path, dehazed.astype(np.uint8))
    print(f"Dehazed image saved to {output_path}")
