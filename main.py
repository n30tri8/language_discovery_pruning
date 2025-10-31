import argparse
import csv
import os
from itertools import product

import torch

from mmlu_evaluation import evaluate_raw_model_on_mmlu, evaluate_pruned_model
from pruning import prepare_calibration
from submodules.SparseLLM.datautils import get_glue
from submodules.SparseLLM.model_utils import llama_sparsellm
from utils import setup_environment, setup_tokenizer, load_raw_model, save_results, save_pruned_model, \
    load_pruned_model, model_dir

SUBJECTS = ["philosophy", "professional_law", "high_school_mathematics", "professional_psychology"]
LANGUAGES = ["EN"]

LINGUISTIC_BENCHMARKS = {
    "EN GLUE": get_glue,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add a simple module-level model variable to replace the argparse --models option
# Candids: "meta-llama/Llama-3.1-8B-Instruct" \ "meta-llama/Llama-3.2-11B-Vision-Instruct" \ "meta-llama/Llama-3.2-3b-Instruct"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def prune(train_num, test_num, sparsity_ratios, run_env):
    results_rows = []

    logs_file = os.path.join(run_env['results_dir'], "pruning_logs.csv")

    # iterate over the single configured model
    # Raw model evaluation
    print(f"\n=== Evaluating RAW model: {MODEL} ===")
    tokenizer = setup_tokenizer(MODEL)
    raw_model = load_raw_model(MODEL)
    model_name = os.path.basename(MODEL)

    raw_accs = evaluate_raw_model_on_mmlu(raw_model, tokenizer, run_env['benchmark_data_dir'], test_num, SUBJECTS,
                                          LANGUAGES)
    results_rows.append(
        ["raw", MODEL, "None", "0%"] + [f"{acc:.4f}" for acc in raw_accs]
    )

    # Cleanup raw model
    raw_model.cpu()
    del raw_model
    torch.cuda.empty_cache()

    # Pruned model evaluation
    # print(f"\n=== Evaluating PRUNED models for: {MODEL} ===")
    # for subject in SUBJECTS:
    #     for lang in LANGUAGES:
    #         for ratio in sparsity_ratios:
    #             print(
    #                 f"\n=== Pruning on subject '{subject}' at {ratio}% sparsity ==="
    #             )
    #
    #             # Initialize new model and tokenizer
    #             base_model = load_raw_model(MODEL)
    #             tokenizer = setup_tokenizer(MODEL)
    #
    #             # Prepare calibration data
    #             trainloader, max_cal_len = prepare_calibration(tokenizer, run_env['benchmark_data_dir'], subject, lang,
    #                                                            train_num, seed)
    #             base_model.seqlen = max_cal_len
    #
    #             # Prune and evaluate
    #             llama_sparsellm(
    #                 base_model, trainloader, torch.device(DEVICE), ratio / 100.0
    #             )
    #
    #             subtask_accs = evaluate_pruned_model(
    #                 base_model,
    #                 tokenizer,
    #                 run_env['benchmark_data_dir'],
    #                 subject,
    #                 lang,
    #                 test_num,
    #                 seed,
    #             )
    #
    #             # Save results with new format
    #             results_rows.append(
    #                 ["pruned", model_name, subject, f"{ratio}%"]
    #                 + [f"{acc:.4f}" for acc in subtask_accs]
    #             )
    #
    #             # Save model
    #             save_path = model_dir(
    #                 run_env['model_dir'], MODEL, subject, lang, ratio
    #             )
    #             save_pruned_model(base_model, 'model_dir_func')
    #             print(f"Saved pruned model to {save_path}")
    #
    #             # Cleanup
    #             base_model.cpu()
    #             del base_model
    #             torch.cuda.empty_cache()

    # Additional evaluation for GLUE tasks
    print(f"\n=== Evaluating PRUNED models on Linguistic benchmarks for: {MODEL} ===")

    for benchmark in LINGUISTIC_BENCHMARKS:
        for ratio in sparsity_ratios:
            print(f"\n=== Pruning on linguistic benchmark '{benchmark}' ===")

            # Initialize new model and tokenizer
            base_model = load_raw_model(MODEL)
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
            # benchmark_results = evaluate_model_on_dataset(base_model, tokenizer, benchmark_data, benchmark)

            # Save results with new format
            # results_rows.append(
            #     ["pruned", model_name, benchmark, f"{ratio}%"]
            #     + [f"{metric:.4f}" for metric in benchmark_results]
            # )

            # todo refactor EN lang
            # Save model
            save_path = model_dir(
                run_env['model_dir'], MODEL, benchmark, "EN", ratio
            )
            save_pruned_model(base_model, save_path)
            print(f"Saved pruned model to {save_path}")

            # Cleanup
            base_model.cpu()
            del base_model
            torch.cuda.empty_cache()

    # Update save_results call with new header
    header = ["type", "name", "pruned_on", "sparsity"] + SUBJECTS + list(LINGUISTIC_BENCHMARKS.keys())
    save_results(logs_file, results_rows, SUBJECTS, header=header)
    print(f"\nAll done! Results saved to '{logs_file}'.")
    print("Rows:", len(results_rows))
    print("Columns:", len(header))


def cross_benchmark_evaluation(test_num, sparsity_ratios, run_env):
    tokenizer = setup_tokenizer(MODEL)

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

    for linguistic_pruned in LINGUISTIC_BENCHMARKS:
        subject_lang_iter = product(SUBJECTS, LANGUAGES)
        for subject, lang in subject_lang_iter:
            load_path = model_dir(
                run_env['model_dir'], MODEL, linguistic_pruned, lang, sparsity_ratios[0]
            )
            pruned_model, _ = load_pruned_model(load_path, device=DEVICE)
            print(f"\n=== Loaded pruned model from {load_path} ===")
            subtask_accs = evaluate_pruned_model(pruned_model, tokenizer, run_env['benchmark_data_dir'], subject, lang,
                                                 test_num)
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
                    sparsity_ratios[0],
                    subject,
                    acc,
                ])
    fout.close()


if __name__ == "__main__":
    is_local = os.environ.get('LOCAL_RUN') is not None
    is_local_docker = os.environ.get('LOCAL_DOCKER_RUN') is not None
    run_env = {}
    if is_local:
        run_env['root_storage_dir'] = os.path.dirname(os.path.abspath(__file__))
        run_env['model_dir'] = os.path.expanduser("~/.cache/huggingface/hub")
    elif is_local_docker:
        run_env['root_storage_dir'] = "/app/dev_root"
        run_env['model_dir'] = "/app/dev_hf_cache"
    else:
        run_env['root_storage_dir'] = "/gcs/language-discovery-pruning/"
        run_env['model_dir'] = os.path.join(run_env['root_storage_dir'], ".cache/huggingface/hub")
    run_env['benchmark_data_dir'] = os.path.join(run_env['root_storage_dir'], "benchmark_data", "mmlu")
    run_env['results_dir'] = os.path.join(run_env['root_storage_dir'], "logs")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_num", type=int, default=32, help="Calibration set size per subtask."
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=20,
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
    args = parser.parse_args()

    setup_environment(args.seed, run_env['model_dir'])
    prune(args.train_num, args.test_num, args.sparsity_ratios, run_env)
    cross_benchmark_evaluation(args.test_num, args.sparsity_ratios, run_env)

