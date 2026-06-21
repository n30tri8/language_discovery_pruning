# AGENTS.md

## What this repo does
- `main.py` is the entrypoint for three experiment stages: `raw_eval`, `prune`, and `cross_eval`.
- The project studies Wanda pruning on multilingual LLMs using MMLU for general knowledge and language-specific benchmarks for pruning/evaluation.

## Core flow to understand first
- `setup_environment()` in `utils.py` seeds `random`, `numpy`, and `torch`, logs into Hugging Face with `HF_TOKEN`, and rewires the raw model cache path.
- `apply_benchmark_dir()` in `main.py` mutates `LINGUISTIC_BENCHMARKS` in-place so each loader/spec gets the correct dataset root.
- `prune()` loads a raw model, builds calibration batches, calls `prepare_calibration()` then `prune_wanda()`, evaluates the pruned model, and optionally saves it in a background thread.

## Data/model conventions
- MMLU data lives in `benchmark_data/mmlu/{lang}.csv`; `get_mmlu()` expects rows shaped like `id, question, A, B, C, D, answer, subject`.
- XGLUE data is directory-driven under `benchmark_data/xglue_dataset/`; Italian, Arabic, and Hindi use custom loaders plus language-specific eval specs.
- Prompt templates are defined in `benchmark_loader/*prompt_templates.py` and must keep chat-style `{field}` / `{premise}` / `{sentence1}` placeholders.

## Evaluation contracts
- All linguistic benchmarks inherit from `evaluation/common_evaluation.py::EvalSpec` and must implement `load_eval_data()` and `extract_answer()`.
- `evaluate_on_linguistic()` builds system/user chat messages, calls `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, then parses the generated text via `extract_answer()`.
- Answer extraction is label-driven and conservative: bracketed forms like `[entailment]`, `[1]`, or last-seen label characters are preferred.

## Pruning implementation boundaries
- Wanda logic lives in `submodules/wanda/prune.py`; `prepare_calibration()` captures layer inputs, and `prune_wanda()` zeros weights by sparsity ratio.
- Pruned models are named by `utils.model_dir()` as `models--<basename>_<benchmark>_<lang>_<ratio>pct`; keep this naming stable because `cross_benchmark_evaluation()` loads from it.
- The code aggressively frees memory with `gc.collect()` and `torch.cuda.empty_cache()` after each major step; preserve that pattern when adding loops.

## Runtime/workflow notes
- `RUN_ENV` controls storage paths: `local`, `local_docker`, `prod_os`, and `google_cloud` are the supported branches in `main.py`.
- The README’s canonical local run is `python main.py --model <hf-model> --run raw_eval prune cross_eval --languages en`.
- There is no repository test suite; use small smoke runs or the `test_*` helpers in `benchmark_loader/datautils.py` when validating data plumbing.

## When editing
- Prefer changing loader/spec/template code over Wanda internals unless the pruning algorithm itself is the target.
- If you add a language, update `LINGUISTIC_BENCHMARKS`, `apply_benchmark_dir()`, a loader in `benchmark_loader/datautils.py`, and a matching `EvalSpec` subclass.
