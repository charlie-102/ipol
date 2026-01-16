# Changelog (Detailed)

## [Unreleased]

### Added
- Committed 22 new method adapters: 15 IPOL 2023 + 7 preprints #ipol #git (2026-01-16) [#ffda]
  - IPOL 2023: bsde_denoising, video_noise, epr_reconstruction, mprnet, mesh_compression, burst_superres, federated_learning, ganet, chromatic_aberration, segmentation_zoo, opencco, signal_decomposition, monocular_depth, homography, shape_vectorization
  - Preprints: fpn_reduction, spherical_splines, emvd_video_denoising, image_matting, siamte, voronoi_segmentation, bsde_segmentation
- `plan_load.sh` hook script for session start context display #session-memory #workflow (2026-01-15) [#eabb]
  - Files: `~/.claude/scripts/plan_load.sh`
- `.claude/plan.md` persistent project memory template #session-memory (2026-01-15) [#eabb]
  - Files: `.claude/plan.md`
- 16 IPOL 2024 method adapters: armcoda, bigcolor, dark_channel, domain_seg, icolorit, image_abstraction, interactive_seg, line_segment, nerf_vaxnerf, noisesniffer, phinet, slavc, storm, superpixel_color, survival_forest, tsne #ipol #2024 #methods (2026-01-15) [#86aa]
  - 8 CPU-compatible: dark_channel, image_abstraction, line_segment, noisesniffer, storm, tsne, armcoda, survival_forest
  - 8 CUDA-required: phinet, bigcolor, icolorit, superpixel_color, interactive_seg, domain_seg, nerf_vaxnerf, slavc
- PyTorch implementation of PhiNet with Keras HDF5 weight converter #pytorch #backend-conversion (2026-01-15) [#86aa]
  - Files: `ipol_runner/methods/phinet_pytorch.py`
- Pure Python backends for dark_channel (dark channel prior dehazing) and image_abstraction (topographic abstraction) #python #backend-conversion (2026-01-15) [#86aa]
  - Files: `ipol_runner/methods/dark_channel_python.py`, `ipol_runner/methods/image_abstraction_python.py`
- Complete `ipol_runner` CLI package (list, run, test, status, deps, compare, gallery) #ipol #cli (2026-01-15) [#b69f]
  - Files: `ipol_runner/cli.py`, `ipol_runner/__main__.py`, `ipol_runner/base.py`
- 10 method adapters for 2025 IPOL publications #methods (2026-01-15) [#b69f]
  - 6 passing: qmsanet, kervrann, cstrd, phase_unwrap, nerf_specularity, semiogram
  - 2 CUDA-only: gaussian_splatting, latent_diffusion
  - 2 Docker-only: sign_lmsls, sign_asslisu
- Validation and testing framework with JSON status tracking #testing (2026-01-15) [#b69f]
  - Files: `ipol_runner/validation.py`, `ipol_runner/testing.py`
- Conda env setup scripts for dependency-conflicting methods #infrastructure (2026-01-15) [#b69f]
  - Files: `scripts/envs/setup_cstrd.sh`, `scripts/envs/setup_kervrann.sh`
- iio shim module wrapping imageio.v3 as iio-compatible API #infrastructure (2026-01-15) [#b69f]
  - Files: `scripts/envs/iio_shim.py`
- Documentation: README, ADDING_METHODS guide, TODO roadmap #docs (2026-01-15) [#b69f]

### Fixed
- .gitignore pattern using `/methods/` instead of `methods/` to avoid catching `ipol_runner/methods/` #bugfix (2026-01-15) [#b69f]

### Changed
- Methods with CUDA/Docker requirements skip gracefully via `requires_cuda`/`requires_docker` properties #infrastructure (2026-01-15) [#b69f]

### Changed
- `/done` skill updated with Step 6.5 for auto-sync to plan.md #workflow (2026-01-15) [#eabb]
  - Files: `~/.claude/skills/done/SKILL.md`
- `~/.claude/CLAUDE.md` updated with Session Memory Protocol section #workflow (2026-01-15) [#eabb]
- `~/.claude/settings.json` updated with plan_load.sh hook #hooks (2026-01-15) [#eabb]
- docs/ADDING_METHODS.md expanded with comprehensive guide: method types, TF→PyTorch patterns, C++→Python patterns, common issues table #docs (2026-01-15) [#86aa]
- README.md updated with method classification tables by category and year #docs (2026-01-15) [#86aa]
- TODO.md 2024 section completed with all 16 methods marked done #planning (2026-01-15) [#86aa]

### Learned
- **Decision**: plan.md is per-project only, no global fallback - clean separation between projects #session-memory #architecture (2026-01-15) [#eabb]
- **Decision**: Auto-create plan.md on first /done, not session start - no upfront setup required #session-memory #workflow (2026-01-15) [#eabb]
- **Pattern**: Documentation separation - TODO.md = backlog (human), plan.md = active (Claude), CHANGELOG = history #workflow #docs (2026-01-15) [#eabb]
- **Pattern**: Hook order on session start: session_check → knowledge_load → plan_load #hooks #workflow (2026-01-15) [#eabb]
- **Gotcha**: ConvTranspose2d weight conversion from Keras requires additional transpose `(1, 0, 2, 3)` after standard Conv2d transpose `(3, 2, 0, 1)` - PyTorch ConvTranspose2d expects (in_channels, out_channels, H, W) not (out_channels, in_channels, H, W) #pytorch #weights (2026-01-15) [#86aa]
  - Files: `ipol_runner/methods/phinet_pytorch.py:208-209`
- **Pattern**: Dual-backend adapter pattern - expose `backend` parameter with choices `["python", "cpp"]` for methods originally in C++ but converted to Python for portability #backend-conversion #architecture (2026-01-15) [#86aa]
  - Files: `ipol_runner/methods/dark_channel.py`, `ipol_runner/methods/image_abstraction.py`
- **Decision**: Convert legacy TensorFlow/Keras to PyTorch rather than maintaining TF compatibility - modern PyTorch is more portable, better maintained, and avoids TF1.x `tensorflow.contrib` issues #pytorch #architecture (2026-01-15) [#86aa]
- **Gotcha**: BatchNorm weight conversion must include all 4 parameters: weight (gamma), bias (beta), running_mean, running_var - missing running stats causes silent failures #pytorch #weights (2026-01-15) [#86aa]
- **Gotcha**: shapely 2.x breaks LineString subclassing with error `LineString.__new__() takes from 1 to 2 positional arguments` - use `shapely<2` for older code that subclasses geometry objects #methods #dependencies (2026-01-15) [#b69f]
  - Files: `scripts/envs/setup_cstrd.sh`
- **Gotcha**: numba requires numpy<2.4 - version constraint critical for JIT compilation compatibility #methods #dependencies (2026-01-15) [#b69f]
  - Files: `scripts/envs/setup_kervrann.sh`
- **Pattern**: iio shim pattern - wrap imageio.v3 as `iio.py` in site-packages to provide iio-compatible API (`iio.read()`, `iio.write()`) when original iio package unavailable #methods #compatibility (2026-01-15) [#b69f]
  - Files: `scripts/envs/iio_shim.py`
- **Decision**: Mark methods `requires_cuda`/`requires_docker` to skip gracefully vs fail - allows test suite to run on any machine while clearly indicating which methods need special infrastructure #testing #architecture (2026-01-15) [#b69f]
  - Files: `ipol_runner/base.py`, `ipol_runner/testing.py`
- **Gotcha**: .gitignore `methods/` pattern catches any path containing `methods/` - use `/methods/` for root-only matching to avoid excluding `ipol_runner/methods/` #git (2026-01-15) [#b69f]
  - Files: `.gitignore`
- **Pattern**: Delete ZIP archives after extraction - downloaded ZIPs remain redundant after extraction, 35 files totaling 429MB in this project #cleanup #storage (2026-01-16) [#ffda]
- **Gotcha**: HEREDOC in git commit fails in Claude sandbox ("can't create temp file for here document") - use direct quoted strings instead #git #sandbox (2026-01-16) [#ffda]
