# Repository Guidelines

## Project Structure & Module Organization
AutoTimes adapts decoder-only LLMs for forecasting through modular Python packages.
- `run.py` selects the experiment class under `exp/` (long/short/zero-shot/in-context tasks).
- `models/`, `layers/`, and `utils/` house the forecasting blocks, adapters, and helpers shared by every script.
- `data_provider/` loads datasets placed under `dataset/`; create this folder locally alongside downloaded `.pt` timestamp embeddings.
- Recipes live in `scripts/<task>/<setting>/AutoTimes_<Dataset>.sh`, while `predict.ipynb` documents an interactive workflow.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` — install PyTorch, transformers, and utilities inside a clean Python ≥3.9 environment.
- `python preprocess.py --gpu 0 --dataset ETTh1` — regenerate textual time-stamp embeddings whenever you add data.
- `python run.py --task_name long_term_forecast --is_training 1 --model_id etth1_llama --model AutoTimes_Llama --data ETTh1 --seq_len 672 --label_len 576 --token_len 96` — run training with explicit hyperparameters.
- `bash scripts/time_series_forecasting/long_term/AutoTimes_ETTh1.sh` or `bash scripts/zero_shot_forecasting/sM4_tM3.sh` — execute the vetted recipes for benchmarking; keep prerequisite checkpoints (e.g., short-term models) in place.

## Coding Style & Naming Conventions
Use 4-space indentation, PEP 8 naming, and type hints where they clarify tensor shapes. Classes stay `PascalCase`, functions/configs `snake_case`, and CLI flags must mirror `run.py` arguments. Keep shell scripts executable, match the `AutoTimes_<Dataset>.sh` pattern, and document non-default arguments at the top of the script.

## Testing Guidelines
There is no unit-test harness; validation happens through the task scripts. When touching modeling, data, or preprocessing code, rerun the relevant script, record SMAPE/MSE reported by `exp/*`, and place the artifact under `test/<run_name>`. Regenerate embeddings after data tweaks, reuse the default seed (`fix_seed=2021`), and note any intentional deviations in your PR.

## Commit & Pull Request Guidelines
The distributed snapshot lacks Git history, so follow a Conventional-Commit-style, imperative subject (`feat: add opt 1b schedule`, `fix: guard amp init`). Describe affected tasks, datasets, and hyperparameters in the body plus the command used to verify changes. Pull requests should include a summary, reproduction commands, metric deltas or log excerpts, and confirmation that large checkpoints/datasets stay out of Git.

## Security & Configuration Tips
Keep Hugging Face tokens and dataset archives outside the repo; point to them via env vars such as `HF_HOME` or `TRANSFORMERS_CACHE`. Store downloaded LLM weights beside the project (for example `../llama/`) and reference them with `--llm_ckp_dir`. Never commit `.env`, raw data, or credential files; prefer naming overrides via `--des` to share configuration context safely.
