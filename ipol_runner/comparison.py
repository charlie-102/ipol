"""Comparison tools for IPOL methods."""
from pathlib import Path
from typing import List, Optional
import numpy as np

from .registry import get_method
from .runner import run_method


def create_comparison(
    method_names: List[str],
    inputs: List[Path],
    output_dir: Path,
    verbose: bool = False
) -> bool:
    """Run multiple methods and create side-by-side comparison.

    Args:
        method_names: List of method names to compare
        inputs: Input files
        output_dir: Output directory
        verbose: Verbose output

    Returns:
        True if successful
    """
    try:
        import cv2
    except ImportError:
        print("Error: opencv-python required for comparison. Install with: pip install opencv-python")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    images = []
    labels = []

    # Run each method
    for name in method_names:
        method = get_method(name)
        if not method:
            print(f"Warning: Unknown method '{name}', skipping")
            continue

        method_output = output_dir / name
        result = run_method(method, inputs, method_output, {}, verbose=verbose)
        results[name] = result

        if result.success and result.primary_output and result.primary_output.exists():
            img = cv2.imread(str(result.primary_output))
            if img is not None:
                images.append(img)
                labels.append(name)
        else:
            print(f"Warning: {name} failed or no output: {result.error_message}")

    if not images:
        print("Error: No successful outputs to compare")
        return False

    # Create side-by-side comparison
    comparison_path = output_dir / "comparison.png"
    _create_side_by_side(images, labels, comparison_path)
    print(f"Side-by-side comparison: {comparison_path}")

    return True


def _create_side_by_side(
    images: List[np.ndarray],
    labels: List[str],
    output_path: Path,
    max_cols: int = 3
) -> None:
    """Create a grid of images with labels."""
    import cv2

    if not images:
        return

    # Find max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)

    # Label bar height
    label_h = 40

    # Resize all images and add labels
    labeled = []
    for img, label in zip(images, labels):
        # Resize to max dimensions (maintain aspect ratio with padding)
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Create canvas with padding
        canvas = np.zeros((max_h + label_h, max_w, 3), dtype=np.uint8)
        # Center the image
        y_off = label_h + (max_h - new_h) // 2
        x_off = (max_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # Add label
        cv2.putText(
            canvas, label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 255), 2
        )
        labeled.append(canvas)

    # Arrange in grid
    n = len(labeled)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    cell_h, cell_w = labeled[0].shape[:2]
    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, img in enumerate(labeled):
        r, c = i // cols, i % cols
        grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = img

    cv2.imwrite(str(output_path), grid)
