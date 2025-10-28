import argparse
import csv
import os
from itertools import product

import torch

from evaluation import evaluate_raw_model_on_mmlu, evaluate_model_on_dataset, evaluate_pruned_model
from pruning import prepare_calibration, save_pruned_model, load_pruned_model
from submodules.SparseLLM.datautils import get_glue
from submodules.SparseLLM.model_utils import llama_sparsellm
from utils import setup_environment, setup_tokenizer, load_model, save_results

SUBJECTS = ["philosophy"]
LANGUAGES = ["EN"]

LINGUISTIC_BENCHMARKS = {
    "EN GLUE": get_glue,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CSV_NAME = "output.csv"

# Add a simple module-level model variable to replace the argparse --models option
MODEL = "meta-llama/Llama-3.2-3b-Instruct"


def prune():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_num", type=int, default=32, help="Calibration set size per subtask."
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=5,
        help="Test set size per subtask, if negative use all.",
    )
    parser.add_argument(
        "--sparsity_ratios",
        nargs="+",
        type=float,
        # default=[25, 50, 75],
        default=[50],
        help="List of integer percentages for unstructured pruning, e.g. 25 50 75.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_environment(args.seed)
    results_rows = []

    # iterate over the single configured model
    # Raw model evaluation
    print(f"\n=== Evaluating RAW model: {MODEL} ===")
    tokenizer = setup_tokenizer(MODEL)
    raw_model = load_model(MODEL)
    model_name = os.path.basename(MODEL)

    raw_accs = evaluate_raw_model_on_mmlu(raw_model, tokenizer, args.test_num, args.seed, SUBJECTS, LANGUAGES)
    results_rows.append(
        ["raw", MODEL, "None", "0%"] + [f"{acc:.4f}" for acc in raw_accs]
    )

    # Cleanup raw model
    raw_model.cpu()
    del raw_model
    torch.cuda.empty_cache()

    # Pruned model evaluation
    print(f"\n=== Evaluating PRUNED models for: {MODEL} ===")
    for subject in SUBJECTS:
        for lang in LANGUAGES:
            for ratio in args.sparsity_ratios:
                print(
                    f"\n=== Pruning on subject '{subject}' at {ratio}% sparsity ==="
                )

                # Initialize new model and tokenizer
                base_model = load_model(MODEL)
                tokenizer = setup_tokenizer(MODEL)

                # Prepare calibration data
                trainloader, max_cal_len = prepare_calibration(tokenizer, subject, lang, args.train_num, args.seed)
                base_model.seqlen = max_cal_len

                # Prune and evaluate
                llama_sparsellm(
                    base_model, trainloader, torch.device(DEVICE), ratio / 100.0
                )

                subtask_accs = evaluate_pruned_model(
                    base_model,
                    tokenizer,
                    subject,
                    lang,
                    args.test_num,
                    args.seed,
                )

                # Save results with new format
                results_rows.append(
                    ["pruned", model_name, subject, f"{ratio}%"]
                    + [f"{acc:.4f}" for acc in subtask_accs]
                )

                # Save model
                save_path = save_pruned_model(
                    base_model, subject, lang, ratio, MODEL
                )
                print(f"Saved pruned model to {save_path}")

                # Cleanup
                base_model.cpu()
                del base_model
                torch.cuda.empty_cache()

    # Additional evaluation for GLUE tasks
    print(f"\n=== Evaluating PRUNED models on Linguistic benchmarks for: {MODEL} ===")

    for benchmark in LINGUISTIC_BENCHMARKS:
        for ratio in args.sparsity_ratios:
            print(f"\n=== Pruning on linguistic benchmark '{benchmark}' ===")

            # Initialize new model and tokenizer
            base_model = load_model(MODEL)
            tokenizer = setup_tokenizer(MODEL)

            # Prepare data
            benchmark_loader = LINGUISTIC_BENCHMARKS[benchmark]
            benchmark_data, max_cal_len = benchmark_loader(tokenizer)

            base_model.seqlen = max_cal_len

            # Prune and evaluate
            llama_sparsellm(
                base_model, benchmark_data, torch.device(DEVICE), ratio / 100.0
            )

            # TODO evaluation differs, do it later
            # Evaluate pruned model on GLUE benchmark
            benchmark_results = evaluate_model_on_dataset(base_model, tokenizer, benchmark_data, benchmark)

            # Save results with new format
            # results_rows.append(
            #     ["pruned", model_name, benchmark, f"{ratio}%"]
            #     + [f"{metric:.4f}" for metric in benchmark_results]
            # )

            # todo refactor EN lang
            # Save model
            save_path = save_pruned_model(
                base_model, benchmark, "EN", ratio, MODEL
            )
            print(f"Saved pruned model to {save_path}")

            # Cleanup
            base_model.cpu()
            del base_model
            torch.cuda.empty_cache()

    # Update save_results call with new header
    header = ["type", "name", "pruned_on", "sparsity"] + SUBJECTS + list(LINGUISTIC_BENCHMARKS.keys())
    save_results(CSV_NAME, results_rows, SUBJECTS, header=header)
    print(f"\nAll done! Results saved to '{CSV_NAME}'.")
    print("Rows:", len(results_rows))
    print("Columns:", len(header))


def cross_benchmark_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_num",
        type=int,
        default=5,
        help="Test set size per subtask, if negative use all.",
    )
    parser.add_argument(
        "--sparsity_ratios",
        nargs="+",
        type=float,
        # default=[25, 50, 75],
        default=[50],
        help="List of integer percentages for unstructured pruning, e.g. 25 50 75.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_environment(args.seed)
    tokenizer = setup_tokenizer(MODEL)

    results_file = "cross_benchmark_results.csv"
    write_header = not os.path.exists(results_file)
    fout = open(results_file, "a", newline="", encoding="utf-8")
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
    for linguistic_pruned in LINGUISTIC_BENCHMARKS:
        subject_lang_iter = product(SUBJECTS, LANGUAGES)
        for subject, lang in subject_lang_iter:
            pruned_model, load_path = load_pruned_model(
                MODEL, linguistic_pruned, lang, args.sparsity_ratios[0], device=DEVICE
            )
            print(f"\n=== Loaded pruned model from {load_path} ===")
            subtask_accs = evaluate_pruned_model(
                pruned_model,
                tokenizer,
                subject,
                lang,
                args.test_num,
                args.seed,
            )
            print(
                f"Evaluation results on subject '{subject}' and language '{lang}': "
                + ", ".join([f"{acc:.4f}" for acc in subtask_accs])
            )
            # Write results to file
            for acc in subtask_accs:
                writer.writerow([
                    MODEL,
                    linguistic_pruned,
                    lang,
                    args.sparsity_ratios[0],
                    subject,
                    acc,
                ])
    fout.close()


if __name__ == "__main__":
    # prune()
    cross_benchmark_evaluation()
