# Repository Guidelines

## Project Structure & Module Organization
ReconDrive is a Python research codebase for 4D Gaussian Splatting from autonomous driving scenes. Core model code lives in `models/`, with Gaussian autoencoder components in `models/gaussian_autoencoder/`. Dataset loading, nuScenes wrappers, augmentations, and masks are under `dataset/`. Training and inference entry points are in `scripts/`; reusable helpers are in `utils/`. Experiment YAMLs are in `configs/nuscenes/`, longer setup and workflow notes are in `docs/`, images are in `assets/`, and generated outputs/checkpoints should stay in `work_dirs/` or `checkpoints/` rather than source folders.

## Build, Test, and Development Commands
- `bash scripts/install_deps.sh`: install project dependencies where supported; otherwise follow `docs/INSTALL.md` for manual CUDA/PyTorch3D/gsplat/SAM2 setup.
- `bash scripts/train.sh 8 ./configs/nuscenes/recondrive.yaml`: launch multi-GPU training; adjust the first argument to the GPU count.
- `python -m scripts.trainer --cfg_path=./configs/nuscenes/recondrive.yaml --train_4d --devices=1`: run training directly for debugging.
- `bash scripts/inference.sh`: run the default single-GPU inference workflow.
- `python scripts/test_stage1_training.py` or `python scripts/test_ae_training.py`: run lightweight training smoke tests.

## Coding Style & Naming Conventions
Use Python 3.10-compatible code and 4-space indentation. Follow existing PyTorch/PyTorch Lightning patterns: modules and files use `snake_case`, classes use `PascalCase`, and config keys use lower-case YAML names. Keep model-specific logic in `models/`, data transformations in `dataset/`, and CLI orchestration in `scripts/`. Prefer explicit tensor shape comments or assertions around complex geometry, rendering, and voxelization code.

## Testing Guidelines
There is no central pytest suite in this snapshot; tests are script-based smoke checks in `scripts/test_*.py`. Name new checks `test_<component>_<scenario>.py` and keep them runnable as standalone Python scripts. For model or data changes, validate at least one minimal forward/training pass and document required checkpoint or dataset assumptions in the file header or PR notes.

## Commit & Pull Request Guidelines
Recent commits use short, lower-case, imperative summaries such as `fix dense depth supervise bug` and `optimize voxel_gs_head forward structure`; keep subjects concise and focused. Pull requests should include a brief problem statement, the changed config or command path, expected dataset/checkpoint requirements, and before/after metrics or screenshots for rendering changes. Link related issues or experiments, and note any skipped tests with the reason.

## Security & Configuration Tips
Do not commit downloaded nuScenes data, large checkpoints, generated renders, core dumps, or local cache directories. Keep machine-specific paths in YAML overrides or local scripts, and prefer relative paths such as `./data/nuscenes/` when adding shared configs.
