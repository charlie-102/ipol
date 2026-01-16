"""Device utilities for Apple M2/MPS and CUDA support.

This module provides explicit device selection for PyTorch-based methods.
Device selection is EXPLICIT only - user must specify --device parameter.
No auto-detection to ensure reproducible behavior.
"""
from typing import Tuple, List, Optional


# Valid device names
VALID_DEVICES = ["cuda", "mps", "cpu"]


def get_device(device_name: str):
    """Get PyTorch device from explicit user choice.

    No auto-detection - user must explicitly specify device.

    Args:
        device_name: One of "cuda", "mps", or "cpu"

    Returns:
        torch.device object

    Raises:
        ValueError: If device_name is not valid
        RuntimeError: If requested device is not available
    """
    import torch

    if device_name not in VALID_DEVICES:
        raise ValueError(f"Invalid device: {device_name}. Choose from: {VALID_DEVICES}")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if device_name == "mps":
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available (requires Apple Silicon)")
        return torch.device("mps")

    return torch.device("cpu")


def check_device_available(device_name: str) -> Tuple[bool, str]:
    """Check if a device is available.

    Args:
        device_name: One of "cuda", "mps", or "cpu"

    Returns:
        Tuple of (available: bool, message: str)
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch not installed"

    if device_name == "cuda":
        avail = torch.cuda.is_available()
        if avail:
            gpu_name = torch.cuda.get_device_name(0)
            return True, f"CUDA available ({gpu_name})"
        return False, "CUDA not available"

    if device_name == "mps":
        if not hasattr(torch.backends, 'mps'):
            return False, "MPS backend not in this PyTorch version"
        avail = torch.backends.mps.is_available()
        if avail:
            return True, "MPS available (Apple Silicon)"
        return False, "MPS not available (requires Apple Silicon)"

    return True, "CPU always available"


def get_available_devices() -> List[str]:
    """Get list of available devices on this system.

    Returns:
        List of available device names
    """
    available = ["cpu"]  # CPU always available

    try:
        import torch

        if torch.cuda.is_available():
            available.append("cuda")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available.append("mps")
    except ImportError:
        pass

    return available


def get_recommended_device() -> str:
    """Get recommended device for this system.

    Preference: cuda > mps > cpu

    Returns:
        Recommended device name
    """
    available = get_available_devices()

    if "cuda" in available:
        return "cuda"
    if "mps" in available:
        return "mps"
    return "cpu"


def print_device_info():
    """Print information about available devices."""
    print("Device Information:")
    print("-" * 40)

    for device in VALID_DEVICES:
        avail, msg = check_device_available(device)
        status = "[OK]" if avail else "[--]"
        print(f"  {status} {device}: {msg}")

    print()
    print(f"Recommended device: {get_recommended_device()}")


def convert_cuda_to_device(code: str, device_var: str = "device") -> str:
    """Convert CUDA-specific code patterns to device-agnostic code.

    Common patterns:
    - .cuda() -> .to(device)
    - torch.cuda.FloatTensor -> torch.FloatTensor(...).to(device)
    - if torch.cuda.is_available() -> if device.type != 'cpu'

    Args:
        code: Python code string
        device_var: Name of the device variable

    Returns:
        Converted code string
    """
    import re

    # .cuda() -> .to(device)
    code = re.sub(r'\.cuda\(\)', f'.to({device_var})', code)

    # torch.cuda.amp.autocast() -> torch.amp.autocast(device_type=device.type)
    code = re.sub(
        r'torch\.cuda\.amp\.autocast\(\)',
        f'torch.amp.autocast(device_type={device_var}.type)',
        code
    )

    return code


class DeviceContext:
    """Context manager for device-specific execution.

    Example:
        with DeviceContext("mps") as ctx:
            model = model.to(ctx.device)
            output = model(input.to(ctx.device))
    """

    def __init__(self, device_name: str):
        """Initialize device context.

        Args:
            device_name: One of "cuda", "mps", or "cpu"
        """
        self.device_name = device_name
        self.device = None

    def __enter__(self):
        self.device = get_device(self.device_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        if self.device_name == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        return False


def setup_environment_for_device(device_name: str):
    """Setup environment variables for a specific device.

    Args:
        device_name: One of "cuda", "mps", or "cpu"
    """
    import os

    if device_name == "cpu":
        # Limit CPU threads to avoid OMP issues
        os.environ.setdefault("OMP_NUM_THREADS", "4")

    if device_name == "mps":
        # MPS-specific settings
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Set matplotlib config dir to avoid permission issues
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/claude/matplotlib")


# =============================================================================
# ONE-CLICK SWITCH HELPERS
# =============================================================================
# These functions make it easy to convert CUDA-only code to device-agnostic code

def set_random_seeds(seed: int, device_name: str = "cpu"):
    """Set random seeds for reproducibility across all devices.

    Replaces CUDA-specific seed setting with device-agnostic version.

    Args:
        seed: Random seed value
        device_name: Device type ("cuda", "mps", "cpu")
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)

        if device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        elif device_name == "mps":
            # MPS uses torch.manual_seed for randomness
            pass
    except ImportError:
        pass


def autocast_context(device_name: str, enabled: bool = True):
    """Get autocast context manager for any device.

    Replaces torch.cuda.amp.autocast() with device-agnostic version.

    Args:
        device_name: Device type ("cuda", "mps", "cpu")
        enabled: Whether to enable autocast

    Returns:
        Context manager for mixed precision
    """
    import torch

    if not enabled or device_name == "cpu":
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()

    # PyTorch 2.0+ has unified autocast
    if hasattr(torch, 'autocast'):
        return torch.autocast(device_type=device_name, enabled=enabled)
    elif device_name == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    else:
        from contextlib import nullcontext
        return nullcontext()


def to_device(tensor_or_model, device):
    """Move tensor or model to device safely.

    Handles edge cases like None values and already-on-device tensors.

    Args:
        tensor_or_model: PyTorch tensor, model, or None
        device: torch.device or device name string

    Returns:
        Tensor/model on specified device, or None if input was None
    """
    if tensor_or_model is None:
        return None

    import torch

    if isinstance(device, str):
        device = torch.device(device)

    return tensor_or_model.to(device)


def load_weights_to_device(checkpoint_path: str, device_name: str = "cpu"):
    """Load model weights to specified device.

    Replaces torch.load(path, map_location='cuda') with device-agnostic version.

    Args:
        checkpoint_path: Path to checkpoint file
        device_name: Target device ("cuda", "mps", "cpu")

    Returns:
        Loaded checkpoint dict
    """
    import torch

    if device_name == "cuda":
        map_location = "cuda"
    elif device_name == "mps":
        map_location = "mps"
    else:
        map_location = "cpu"

    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def get_float_tensor_type(device_name: str):
    """Get the appropriate float tensor type for a device.

    Replaces torch.cuda.FloatTensor with device-agnostic version.

    Args:
        device_name: Device type

    Returns:
        Tensor class to use
    """
    import torch

    # For MPS and CPU, we create regular tensors then move to device
    # Only CUDA has specific tensor types
    if device_name == "cuda":
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor


def create_tensor_on_device(data, device_name: str, dtype=None):
    """Create a tensor directly on the specified device.

    Args:
        data: Input data (list, numpy array, etc.)
        device_name: Target device
        dtype: Optional dtype (default: float32)

    Returns:
        Tensor on specified device
    """
    import torch

    if dtype is None:
        dtype = torch.float32

    device = get_device(device_name)
    return torch.tensor(data, dtype=dtype, device=device)


def patch_module_for_device(module, device_name: str):
    """Patch a module to be device-agnostic.

    Monkey-patches common CUDA-specific attributes and methods.
    Use this to quickly convert CUDA-only code.

    Args:
        module: Python module to patch
        device_name: Target device name
    """
    import torch

    device = get_device(device_name)

    # Store device for module-level access
    module._device = device
    module._device_name = device_name

    # Patch common functions if they exist
    if hasattr(module, 'cuda'):
        original_cuda = module.cuda
        module.cuda = lambda x=None: to_device(x, device) if x is not None else device


# =============================================================================
# METHOD-SPECIFIC PATCHES
# =============================================================================

def get_bigcolor_patches(device_name: str) -> dict:
    """Get patches needed for bigcolor method.

    Returns dict of {function_name: replacement_function}
    """
    import torch
    device = get_device(device_name)

    def patched_set_seed(seed):
        set_random_seeds(seed, device_name)

    def patched_seed_rng(rng, seed=0):
        set_random_seeds(seed, device_name)
        return rng

    def patched_load_weights(weight_path):
        return load_weights_to_device(str(weight_path), device_name)

    return {
        'set_seed': patched_set_seed,
        'seed_rng': patched_seed_rng,
        'load_weights': patched_load_weights,
        'device': device,
    }


def get_icolorit_patches(device_name: str) -> dict:
    """Get patches needed for icolorit method."""
    import torch
    device = get_device(device_name)

    def patched_autocast(enabled=True):
        return autocast_context(device_name, enabled)

    return {
        'autocast': patched_autocast,
        'device': device,
    }


def get_latent_diffusion_patches(device_name: str) -> dict:
    """Get patches needed for latent_diffusion method."""
    import torch
    device = get_device(device_name)

    return {
        'device': device,
        'device_name': device_name,
    }


if __name__ == "__main__":
    print_device_info()
