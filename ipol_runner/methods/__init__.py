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

# IPOL Preprint methods
from . import voronoi_segmentation # pre-591 - Voronoi Page Segmentation
from . import bsde_segmentation    # pre-636 - BSDE Image Segmentation
from . import siamte               # pre-558 - SiamTE Camera Trace Extraction
from . import image_matting        # pre-532 - Natural Image Matting
from . import emvd_video_denoising # pre-464 - Multi-Stage Video Denoising
from . import fpn_reduction        # pre-436 - Fixed Pattern Noise Reduction
from . import spherical_splines    # pre-451 - Thin-Plate Splines on Sphere

# IPOL 2023 methods
from . import homography           # 356 - Robust Homography Estimation
from . import shape_vectorization  # 401 - Binary Shape Vectorization
from . import epr_reconstruction   # 414 - EPR Image Reconstruction (TV)
from . import mesh_compression     # 418 - Progressive Mesh Compression
from . import signal_decomposition # 417 - Signal Decomposition (Python)
from . import video_noise          # 420 - Video Noise Estimation
from . import federated_learning   # 440 - One-Shot Federated Learning (FESC)
from . import ganet                # 441 - GANet Stereo Matching
from . import chromatic_aberration # 443 - Chromatic Aberration Correction
from . import mprnet               # 446 - MPRNet Image Restoration
from . import segmentation_zoo     # 447 - Semantic Segmentation Zoo
from . import monocular_depth      # 459 - Monocular Depth Estimation (4 methods)
from . import burst_superres       # 460 - Handheld Burst Super-Resolution
from . import bsde_denoising       # 467 - BSDE Image Denoising
from . import opencco              # 477 - OpenCCO Vascular Tree Generation
