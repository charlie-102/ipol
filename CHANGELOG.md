# Changelog

## 2026-01-15

### Session: Implemented persistent session memory via plan.md [#eabb]
- **Added**: `plan_load.sh` hook script to display project context on session start #session-memory #workflow
  - Files: `~/.claude/scripts/plan_load.sh`
- **Added**: `.claude/plan.md` persistent project memory with focus, tasks, decisions sections #session-memory
  - Files: `.claude/plan.md`
- **Changed**: Updated `/done` skill with Step 6.5 to auto-sync state to plan.md #workflow
  - Files: `~/.claude/skills/done/SKILL.md`
- **Changed**: Updated `~/.claude/CLAUDE.md` with Session Memory Protocol section #workflow
  - Files: `~/.claude/CLAUDE.md`
- **Changed**: Updated `~/.claude/settings.json` to add plan_load.sh to SessionStart hooks #hooks
  - Files: `~/.claude/settings.json`
- **Discussed**: Documentation coordination - TODO.md (backlog) vs plan.md (active) vs CHANGELOG (history) #planning
- **Discussed**: Optimal workflows for feature implementation vs quick tasks #workflow

---

### Session: Added all 16 IPOL 2024 method adapters with backend conversions [#86aa]
- **Added**: 16 IPOL 2024 method adapters #ipol #2024 #methods
  - Files: `ipol_runner/methods/armcoda.py`, `bigcolor.py`, `dark_channel.py`, `domain_seg.py`, `icolorit.py`, `image_abstraction.py`, `interactive_seg.py`, `line_segment.py`, `nerf_vaxnerf.py`, `noisesniffer.py`, `phinet.py`, `slavc.py`, `storm.py`, `superpixel_color.py`, `survival_forest.py`, `tsne.py`
- **Added**: PyTorch implementation of PhiNet (converted from TensorFlow/Keras) with weight conversion utility #pytorch #backend-conversion
  - Files: `ipol_runner/methods/phinet_pytorch.py`
- **Added**: Pure Python backends for dark_channel and image_abstraction (originally C++) #python #backend-conversion
  - Files: `ipol_runner/methods/dark_channel_python.py`, `ipol_runner/methods/image_abstraction_python.py`
- **Changed**: Updated docs/ADDING_METHODS.md with comprehensive guide including TF→PyTorch conversion patterns, C++→Python backend patterns, and lessons learned #docs
- **Changed**: Updated README.md with method classification tables by category and year #docs
- **Changed**: Completed 2024 section in TODO.md #planning

---

### Session: Built IPOL Runner unified CLI with 10 method adapters [#b69f]
- **Added**: Complete `ipol_runner` CLI package with commands: list, run, test, status, deps, compare, gallery #ipol #cli
  - Files: `ipol_runner/cli.py`, `ipol_runner/__main__.py`, `ipol_runner/base.py`, `ipol_runner/registry.py`, `ipol_runner/runner.py`
- **Added**: 10 method adapters for 2025 IPOL publications #methods
  - Files: `ipol_runner/methods/qmsanet.py`, `ipol_runner/methods/kervrann.py`, `ipol_runner/methods/cstrd.py`, `ipol_runner/methods/phase_unwrap.py`, `ipol_runner/methods/nerf_specularity.py`, `ipol_runner/methods/semiogram.py`, `ipol_runner/methods/gaussian_splatting.py`, `ipol_runner/methods/latent_diffusion.py`, `ipol_runner/methods/sign_language.py`
- **Added**: Validation and testing framework with status tracking #testing
  - Files: `ipol_runner/validation.py`, `ipol_runner/testing.py`
- **Added**: Conda environment setup scripts for methods with dependency conflicts #infrastructure
  - Files: `scripts/envs/setup_cstrd.sh`, `scripts/envs/setup_kervrann.sh`, `scripts/envs/iio_shim.py`
- **Added**: Comprehensive documentation #docs
  - Files: `README.md`, `docs/ADDING_METHODS.md`, `TODO.md`, `CLAUDE.md`
- **Fixed**: .gitignore pattern `/methods/` instead of `methods/` to avoid catching `ipol_runner/methods/` #bugfix
- **Changed**: Methods with CUDA/Docker requirements now skip gracefully with `requires_cuda`/`requires_docker` flags #infrastructure

---
