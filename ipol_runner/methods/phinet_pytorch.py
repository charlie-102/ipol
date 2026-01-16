"""PyTorch implementation of PhiNet for InSAR phase denoising.

Based on: F. Sica, G. Gobbi, P. Rizzoli and L. Bruzzone,
"Phi-Net: Deep Residual Learning for InSAR Parameters Estimation,"
IEEE Transactions on Geoscience and Remote Sensing, 2020.
doi: 10.1109/TGRS.2020.3020427

Architecture: Residual U-Net with 4 encoder levels (64->128->256->512 channels)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional
import logging


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Main path: Conv -> BN -> ReLU -> Conv -> BN
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1)

        # Skip connection: 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class PhiNetPyTorch(nn.Module):
    """PyTorch implementation of Phi-Net for InSAR phase denoising."""

    def __init__(self):
        super().__init__()

        # Encoder path
        self.enc1 = ResidualBlock(2, 64)      # Input: 2 channels (real/imag)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512)

        # Decoder path with transposed convolutions
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ResidualBlock(512, 256)   # 256 + 256 skip = 512

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ResidualBlock(256, 128)   # 128 + 128 skip = 256

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ResidualBlock(128, 64)    # 64 + 64 skip = 128

        # Output: 2 channels (phase, coherence)
        self.output = nn.Conv2d(64, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


def convert_keras_to_pytorch(keras_h5_path: Path, pytorch_path: Path) -> None:
    """Convert Keras HDF5 weights to PyTorch state dict."""
    import h5py

    model = PhiNetPyTorch()
    state_dict = model.state_dict()

    # Layer mapping from Keras to PyTorch
    # Keras uses (H, W, C_in, C_out), PyTorch uses (C_out, C_in, H, W)
    layer_mapping = {
        # Encoder 1
        'conv2d_2': 'enc1.conv1',      # 3x3 (2->64)
        'conv2d_3': 'enc1.conv2',      # 3x3 (64->64)
        'conv2d_1': 'enc1.skip',       # 1x1 (2->64)
        'batch_normalization_1': 'enc1.bn1',
        'batch_normalization_2': 'enc1.bn2',

        # Encoder 2
        'conv2d_5': 'enc2.conv1',      # 3x3 (64->128)
        'conv2d_6': 'enc2.conv2',      # 3x3 (128->128)
        'conv2d_4': 'enc2.skip',       # 1x1 (64->128)
        'batch_normalization_3': 'enc2.bn1',
        'batch_normalization_4': 'enc2.bn2',

        # Encoder 3
        'conv2d_8': 'enc3.conv1',      # 3x3 (128->256)
        'conv2d_9': 'enc3.conv2',      # 3x3 (256->256)
        'conv2d_7': 'enc3.skip',       # 1x1 (128->256)
        'batch_normalization_5': 'enc3.bn1',
        'batch_normalization_6': 'enc3.bn2',

        # Bottleneck
        'conv2d_11': 'bottleneck.conv1',  # 3x3 (256->512)
        'conv2d_12': 'bottleneck.conv2',  # 3x3 (512->512)
        'conv2d_10': 'bottleneck.skip',   # 1x1 (256->512)
        'batch_normalization_7': 'bottleneck.bn1',
        'batch_normalization_8': 'bottleneck.bn2',

        # Decoder 3 (upsampling)
        'upsampling_512-256': 'up3',
        'conv2d_14': 'dec3.conv1',     # 3x3 (512->256)
        'conv2d_15': 'dec3.conv2',     # 3x3 (256->256)
        'conv2d_13': 'dec3.skip',      # 1x1 (512->256)
        'batch_normalization_9': 'dec3.bn1',
        'batch_normalization_10': 'dec3.bn2',

        # Decoder 2 (upsampling)
        'upsampling_256-128': 'up2',
        'conv2d_17': 'dec2.conv1',     # 3x3 (256->128)
        'conv2d_18': 'dec2.conv2',     # 3x3 (128->128)
        'conv2d_16': 'dec2.skip',      # 1x1 (256->128)
        'batch_normalization_11': 'dec2.bn1',
        'batch_normalization_12': 'dec2.bn2',

        # Decoder 1 (upsampling)
        'upsampling_128-64': 'up1',
        'conv2d_20': 'dec1.conv1',     # 3x3 (128->64)
        'conv2d_21': 'dec1.conv2',     # 3x3 (64->64)
        'conv2d_19': 'dec1.skip',      # 1x1 (128->64)
        'batch_normalization_13': 'dec1.bn1',
        'batch_normalization_14': 'dec1.bn2',

        # Output
        'conv2d_22': 'output',         # 1x1 (64->2)
    }

    with h5py.File(keras_h5_path, 'r') as f:
        weights = f['model_weights']['model_1']

        for keras_name, pytorch_name in layer_mapping.items():
            if keras_name not in weights:
                logging.warning(f"Layer {keras_name} not found in Keras model")
                continue

            layer_weights = weights[keras_name]

            if 'batch_normalization' in keras_name:
                # BatchNorm: gamma, beta, moving_mean, moving_variance
                gamma = np.array(layer_weights['gamma:0'])
                beta = np.array(layer_weights['beta:0'])
                mean = np.array(layer_weights['moving_mean:0'])
                var = np.array(layer_weights['moving_variance:0'])

                state_dict[f'{pytorch_name}.weight'] = torch.from_numpy(gamma)
                state_dict[f'{pytorch_name}.bias'] = torch.from_numpy(beta)
                state_dict[f'{pytorch_name}.running_mean'] = torch.from_numpy(mean)
                state_dict[f'{pytorch_name}.running_var'] = torch.from_numpy(var)
            else:
                # Conv2D: kernel, bias
                kernel = np.array(layer_weights['kernel:0'])
                bias = np.array(layer_weights['bias:0'])

                # Transpose kernel from (H, W, C_in, C_out) to (C_out, C_in, H, W)
                kernel = np.transpose(kernel, (3, 2, 0, 1))

                # For ConvTranspose2d, PyTorch expects (in_channels, out_channels, H, W)
                # but the converted kernel is (out_channels, in_channels, H, W)
                # So we need to swap the first two dimensions
                if pytorch_name in ['up1', 'up2', 'up3']:
                    kernel = np.transpose(kernel, (1, 0, 2, 3))

                state_dict[f'{pytorch_name}.weight'] = torch.from_numpy(kernel)
                state_dict[f'{pytorch_name}.bias'] = torch.from_numpy(bias)

    model.load_state_dict(state_dict)
    torch.save(state_dict, pytorch_path)
    logging.info(f"Converted weights saved to {pytorch_path}")


class PhiNetInference:
    """High-level inference class for PhiNet."""

    def __init__(self, weights_path: Optional[Path] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PhiNetPyTorch().to(self.device)
        self.model.eval()

        if weights_path and weights_path.exists():
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logging.info(f"Loaded weights from {weights_path}")

    def inference(
        self,
        primary: np.ndarray,
        secondary: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on InSAR image pair.

        Args:
            primary: Complex primary SLC image (H, W)
            secondary: Complex secondary SLC image (H, W)

        Returns:
            phase: Denoised interferometric phase (H, W)
            coherence: Estimated coherence (H, W)
        """
        assert primary.shape == secondary.shape, "Images must have the same size"
        h, w = primary.shape

        # Create interferogram
        intf = primary * np.conj(secondary)

        # Stack real and imaginary as channels
        real = np.real(intf).astype(np.float32)
        imag = np.imag(intf).astype(np.float32)

        # Normalize
        amp = np.sqrt(real**2 + imag**2) + 1e-8
        real_norm = real / amp
        imag_norm = imag / amp

        # Pad to multiple of 8
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8

        if h_pad > 0 or w_pad > 0:
            real_norm = np.pad(real_norm, ((0, h_pad), (0, w_pad)), mode='reflect')
            imag_norm = np.pad(imag_norm, ((0, h_pad), (0, w_pad)), mode='reflect')

        # Create input tensor: (1, 2, H, W)
        x = np.stack([real_norm, imag_norm], axis=0)[np.newaxis, ...]
        x = torch.from_numpy(x).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(x)

        # Extract phase and coherence
        output = output.cpu().numpy()[0]  # (2, H, W)
        phase = output[0, :h, :w]
        coherence = output[1, :h, :w]

        # Coherence should be in [0, 1]
        coherence = np.clip(coherence, 0, 1)

        return phase, coherence

    def apply(self, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
        """Apply PhiNet denoising and return complex interferogram.

        Args:
            primary: Complex primary SLC image
            secondary: Complex secondary SLC image

        Returns:
            Complex denoised interferogram (coherence * exp(1j * phase))
        """
        phase, coherence = self.inference(primary, secondary)
        return coherence * np.exp(1j * phase).astype(np.complex64)


def ensure_pytorch_weights(method_dir: Path) -> Path:
    """Ensure PyTorch weights exist, converting from Keras if needed."""
    pytorch_weights = method_dir / "phinet" / "PhiNet_model.pt"
    keras_weights = method_dir / "phinet" / "PhiNet_model.hdf5"

    if pytorch_weights.exists():
        return pytorch_weights

    if not keras_weights.exists():
        raise FileNotFoundError(f"Keras weights not found: {keras_weights}")

    logging.info("Converting Keras weights to PyTorch format...")
    convert_keras_to_pytorch(keras_weights, pytorch_weights)
    return pytorch_weights


if __name__ == "__main__":
    # Test the model architecture
    model = PhiNetPyTorch()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(1, 2, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
