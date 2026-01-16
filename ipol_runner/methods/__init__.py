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

# IPOL 2024 methods
from . import noisesniffer         # 462 - Forgery Detection
from . import phinet               # 549 - InSAR Phase Denoising
from . import storm                # 496 - STORM Microscopy
from . import bigcolor             # 542 - Image Colorization
from . import icolorit             # 539 - Interactive Colorization
from . import nerf_vaxnerf         # 553 - VaxNeRF Accelerated NeRF
from . import tsne                 # 528 - t-SNE Visualization
from . import armcoda              # 494 - Arm Movement Analysis
from . import dark_channel         # 530 - Dark Channel Dehazing (C++ or Python)
from . import line_segment         # 481 - Line Segment Detection (Multi-method)
from . import image_abstraction    # 495 - Image Abstraction (C++ or Python)
from . import superpixel_color     # 522 - Superpixel Color Transfer
from . import slavc                # 525 - Sound Source Localization
from . import survival_forest      # 466 - LTRC Survival Forest
from . import interactive_seg      # 498 - Interactive Segmentation
from . import domain_seg           # 499 - Domain Generalization Segmentation
