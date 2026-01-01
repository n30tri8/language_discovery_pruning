import argparse
import csv
import os
from functools import partial

import torch

from evaluation.common_evaluation import evaluate_on_linguistic
from evaluation.glue_spec import GlueEvalSpec
from evaluation.it_spec import ITEvalSpec
from evaluation.mmlu_evaluation import evaluate_model
from evaluation.xglue_spec import XGlueEvalSpec
from submodules.SparseLLM.datautils import get_xglue, get_glue, get_italian_calib
from submodules.SparseLLM.model_utils import llama_sparsellm
from utils import setup_environment, setup_tokenizer, load_raw_model, save_results, save_pruned_model_async, \
    load_pruned_model, model_dir

SUBJECTS = ["philosophy", "professional_law", "high_school_mathematics", "professional_psychology"]

LINGUISTIC_BENCHMARKS = {
    "EN GLUE": {
        "lang": "en",
        "loader": get_glue,
        "eval_spec": GlueEvalSpec()
    },
    "XGLUE_DE": {
        "lang": "de",
        "loader": get_xglue,
        "eval_spec": XGlueEvalSpec("XGLUE_DE", "de")
    },
    "XGLUE_FR": {
        "lang": "fr",
        "loader": get_xglue,
        "eval_spec": XGlueEvalSpec("XGLUE_FR", "fr")
    },
    "VARIED_IT": {
        "lang": "it",
        "loader": get_italian_calib,
        "eval_spec": ITEvalSpec("VARIED_IT", "it")
    }
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _normalize_languages(languages):
    unique = []
    seen = set()
    for lang in languages:
        if lang not in seen:
            unique.append(lang)
            seen.add(lang)
    return unique


def evaluate_raw_model(model_name, test_num, run_env, selected_languages):
    results_rows = []

    logs_file = os.path.join(run_env['results_dir'], "raw_model_eval.csv")
    print(f"\n=== Evaluating RAW model: {model_name} ===")
    tokenizer = setup_tokenizer(model_name)
    raw_model = load_raw_model(model_name)
    languages = selected_languages
    for subject in SUBJECTS:
        for lang in languages:
            subtask_acc = evaluate_model(raw_model, tokenizer, run_env['benchmark_data_dir'], subject, lang, test_num)
            results_rows.append([model_name, subject, lang, f"{subtask_acc:.4f}"])

    # Cleanup raw model
    raw_model.cpu()
    del raw_model
    torch.cuda.empty_cache()

    header = ["model", "subject", "lang", "subtask_acc"]
    save_results(logs_file, results_rows, SUBJECTS, header=header)
    print(f"\nRaw model evaluation done. Results saved to '{logs_file}'.")


def prune(model_name, sparsity_ratios, run_env, selected_languages):
    save_threads = []

    tokenizer = setup_tokenizer(model_name)
    allowed_langs = set(selected_languages)

    # Prepare linguistic evaluation logs file
    linguistic_logs_file = os.path.join(run_env['results_dir'], "linguistic_eval_logs.csv")
    write_header = not os.path.exists(linguistic_logs_file)
    os.makedirs(os.path.dirname(linguistic_logs_file), exist_ok=True)
    fout = open(linguistic_logs_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["model_name", "benchmark", "lang", "pruned_ratio", "evaluation_result"])

    for benchmark in LINGUISTIC_BENCHMARKS:
        lang = LINGUISTIC_BENCHMARKS[benchmark]['lang']
        if lang not in allowed_langs:
            continue

        # pre-prune evaluation
        model_to_prune = load_raw_model(model_name)
        evaluation_spec = LINGUISTIC_BENCHMARKS[benchmark]['eval_spec']
        linguistic_eval = evaluate_on_linguistic(model_to_prune, tokenizer, evaluation_spec)
        # Log pre-pruning evaluation (pruned_ratio = 0)
        writer.writerow([
            model_name,
            benchmark,
            lang,
            0,
            str(linguistic_eval),
        ])
        fout.flush()

        print(f"\n=== Pruning on linguistic benchmark '{benchmark}' ===")
        # Prepare data
        benchmark_loader = LINGUISTIC_BENCHMARKS[benchmark]['loader']
        benchmark_data, max_cal_len = benchmark_loader(tokenizer)

        for ratio in sparsity_ratios:
            model_to_prune = load_raw_model(model_name)
            model_to_prune.seqlen = max_cal_len

            # Prune and evaluate
            llama_sparsellm(
                model_to_prune, benchmark_data, max_cal_len, torch.device(DEVICE), ratio / 100.0
            )

            linguistic_eval = evaluate_on_linguistic(model_to_prune, tokenizer, evaluation_spec)

            # Log post-pruning evaluation for this ratio
            writer.writerow([
                model_name,
                benchmark,
                lang,
                ratio,
                str(linguistic_eval),
            ])
            fout.flush()

            # Save model
            save_path = model_dir(
                run_env['model_dir'], model_name, benchmark, lang, ratio
            )
            # Move model to CPU for saving to avoid GPU memory spike during serialization
            model_to_prune.cpu()
            torch.cuda.empty_cache()
            thread = save_pruned_model_async(model_to_prune, save_path)
            save_threads.append(thread)
            print(f"Saving pruned model to {save_path} in a thread: {thread}")

    fout.close()

    for thread in save_threads:
        thread.join()


def cross_benchmark_evaluation(model_name, test_num, sparsity_ratios, run_env, selected_languages):
    tokenizer = setup_tokenizer(model_name)

    logs_file = os.path.join(run_env['results_dir'], "cross_benchmark_logs.csv")
    write_header = not os.path.exists(logs_file)
    fout = open(logs_file, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow([
            "model_name",
            "pruned on",
            "language",
            "sparsity_ratio",
            "benchmark",
            "accuracy",
        ])

    allowed_langs = set(selected_languages)
    for linguistic_pruned in LINGUISTIC_BENCHMARKS:
        lang = LINGUISTIC_BENCHMARKS[linguistic_pruned]['lang']
        if lang not in allowed_langs:
            continue
        load_path = model_dir(
            run_env['model_dir'], model_name, linguistic_pruned, lang, sparsity_ratios[0]
        )
        pruned_model, _ = load_pruned_model(load_path, device=DEVICE)
        print(f"\n=== Loaded pruned model from {load_path} ===")

        for subject in SUBJECTS:
            subtask_acc = evaluate_model(pruned_model, tokenizer, run_env['benchmark_data_dir'], subject, lang,
                                         test_num)
            print(f"Evaluation results on subject '{subject}' and language '{lang}': {subtask_acc:.4f}")
            # Write results to file
            writer.writerow([model_name, linguistic_pruned, lang, sparsity_ratios[0], subject, subtask_acc])
    fout.close()


def apply_benchmark_dir(proj_dir):
    xglue_base_dir = os.path.join(proj_dir, "benchmark_data", "xglue_dataset")
    it_base_dir = os.path.join(proj_dir, "benchmark_data")

    for benchmark in LINGUISTIC_BENCHMARKS:
        lang = LINGUISTIC_BENCHMARKS[benchmark]['lang']
        loader = LINGUISTIC_BENCHMARKS[benchmark]['loader']
        if loader is get_xglue:
            partial_get_xglue = partial(get_xglue, base_dir=xglue_base_dir, lang=lang)
            LINGUISTIC_BENCHMARKS[benchmark]['loader'] = partial_get_xglue
            # also pass the base_dir to EvalSpec class
            LINGUISTIC_BENCHMARKS[benchmark]['eval_spec'].set_dataset_base_dir(xglue_base_dir)
        elif lang == "it":
            partial_get_italian_calib = partial(get_italian_calib, base_dir=it_base_dir)
            LINGUISTIC_BENCHMARKS[benchmark]['loader'] = partial_get_italian_calib
            # also pass the base_dir to EvalSpec class
            LINGUISTIC_BENCHMARKS[benchmark]['eval_spec'].set_dataset_base_dir(it_base_dir)


if __name__ == "__main__":
    is_local = os.environ.get('LOCAL_RUN') is not None
    is_local_docker = os.environ.get('LOCAL_DOCKER_RUN') is not None
    run_env = {}
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if is_local:
        run_env['root_storage_dir'] = os.path.dirname(os.path.abspath(__file__))
        run_env['model_dir'] = os.path.expanduser("~/.cache/huggingface/hub")
    elif is_local_docker:
        run_env['root_storage_dir'] = "/app/dev_root"
        run_env['model_dir'] = "/app/dev_pruned_models"
    else:
        run_env['root_storage_dir'] = "/gcs/language-discovery-pruning/"
        run_env['model_dir'] = os.path.join(run_env['root_storage_dir'], ".cache/huggingface/hub")
    run_env['raw_model_dir'] = os.path.join(project_dir, "raw_model")
    run_env['benchmark_data_dir'] = os.path.join(project_dir, "benchmark_data", "mmlu")
    run_env['results_dir'] = os.path.join(run_env['root_storage_dir'], "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_num",
        type=int,
        default=50,
        help="Test set size per subtask, if negative use all.",
    )
    parser.add_argument(
        "--sparsity_ratios",
        nargs="+",
        type=float,
        default=[50],
        help="List of integer percentages for unstructured pruning, e.g. 25 50 75.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the model to use, e.g., 'meta-llama/Llama-3.1-8B-Instruct'."
    )
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["raw_eval", "prune", "cross_eval"],
        default=["raw_eval"],
        help="Which procedures to run. Choose any of: raw_eval prune cross_eval. Default: raw_eval.",
    )
    AVAILABLE_LANG_CODES = sorted({config["lang"] for config in LINGUISTIC_BENCHMARKS.values()})
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=AVAILABLE_LANG_CODES,
        default=AVAILABLE_LANG_CODES,
        help=f"Subset of language codes to process (default: {', '.join(AVAILABLE_LANG_CODES)}).",
    )
    args = parser.parse_args()
    to_run = set(args.run)

    selected_languages = _normalize_languages(args.languages)
    selected_langs_set = set(selected_languages)
    if not any(cfg['lang'] in selected_langs_set for cfg in LINGUISTIC_BENCHMARKS.values()):
        raise ValueError("No linguistic benchmarks match the provided languages.")

    setup_environment(args.seed, run_env['raw_model_dir'])
    apply_benchmark_dir(project_dir)

    if "raw_eval" in to_run:
        evaluate_raw_model(args.model, args.test_num, run_env, selected_languages)
    if "prune" in to_run:
        prune(args.model, args.sparsity_ratios, run_env, selected_languages)
    if "cross_eval" in to_run:
        cross_benchmark_evaluation(args.model, args.test_num, args.sparsity_ratios, run_env, selected_languages)
