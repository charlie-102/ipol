# IPOL Runner - TODO

## Year-by-Year Method Expansion

### 2025 (Current)
- [x] qmsanet - Denoising
- [x] kervrann - Change detection (needs ipol_kervrann env + iio shim)
- [x] cstrd - Tree ring detection
- [x] phase_unwrap - Phase processing
- [x] nerf_specularity - 3D reconstruction
- [x] semiogram - Medical/gait analysis
- [ ] gaussian_splatting - 3D reconstruction (CUDA)
- [ ] latent_diffusion - Generation (CUDA)
- [ ] sign_lmsls - Segmentation (Docker)
- [ ] sign_asslisu - Segmentation (Docker)

### 2024
- [x] noisesniffer (462) - Forgery detection
- [x] phinet (549) - InSAR phase denoising (converted to PyTorch)
- [x] storm (496) - STORM microscopy super-resolution
- [x] bigcolor (542) - Image colorization (CUDA)
- [x] icolorit (539) - Interactive colorization (CUDA)
- [x] nerf_vaxnerf (553) - VaxNeRF accelerated NeRF (CUDA/JAX)
- [x] tsne (528) - t-SNE dimensionality reduction
- [x] armcoda (494) - Arm movement analysis
- [x] dark_channel (530) - Dark channel dehazing (Python + C++ backends)
- [x] line_segment (481) - Line segment detection (8 algorithms)
- [x] image_abstraction (495) - Image abstraction (Python + C++ backends)
- [x] superpixel_color (522) - Color transfer (CUDA)
- [x] interactive_seg (498) - Interactive segmentation (CUDA)
- [x] domain_seg (499) - Domain segmentation (CUDA)
- [x] survival_forest (466) - Survival analysis (built-in datasets)
- [x] slavc (525) - Audio-visual sound localization (CUDA)

### 2023
- [ ] Browse https://www.ipol.im/pub/art/?y=2023
- [ ] Select methods to add

### 2022
- [ ] Browse https://www.ipol.im/pub/art/?y=2022
- [ ] Select methods to add

### 2021 and earlier
- [ ] Consider significant/popular methods

---

## Documentation

- [x] `docs/ADDING_METHODS.md` - Step-by-step guide with lessons learned
- [ ] Add example outputs for gallery

---

## Infrastructure

- [ ] Test CUDA methods on GPU machine
- [ ] Set up Docker testing environment for sign_* methods
- [ ] CI/CD for automated testing
- [ ] Generate tiny sample datasets for faster debugging/testing

---

## Notes

See `docs/ADDING_METHODS.md` for:
- Common issues & solutions
- Adapter template
- Environment setup patterns
