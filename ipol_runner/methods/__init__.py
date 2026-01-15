"""IPOL method adapters."""
# Import all method adapters to register them

# Original methods (already installed)
from . import qmsanet
from . import kervrann
from . import cstrd

# IPOL 2025 methods
from . import gaussian_splatting   # 566 - Gaussian Splatting
from . import latent_diffusion     # 580 - Latent Diffusion Aerial Imagery
from . import phase_unwrap         # 583 - Phase Unwrapping
from . import nerf_specularity     # 562 - NeRF Specularity
from . import semiogram            # 535 - Gait Analysis
from . import sign_language        # 560 - Sign Language (LMSLS + ASSLiSU)
