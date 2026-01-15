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
- [ ] Browse https://www.ipol.im/pub/art/?y=2024
- [ ] Select methods to add
- [ ] Follow guide in `docs/ADDING_METHODS.md`

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
