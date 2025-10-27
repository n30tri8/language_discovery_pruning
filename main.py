import argparse
import csv
import os
from itertools import product

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from submodules.SparseLLM.datautils import SYSTEM_PROMPT, _build_user_message, get_mmlu, get_glue
from submodules.SparseLLM.model_utils import llama_sparsellm

SUBJECTS = [
    "philosophy",
]
LANGUAGES = ["EN"]

LINGUISTIC_BENCHMARKS = {
    "EN GLUE": get_glue,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_NAME = "output.csv"

# Add a simple module-level model variable to replace the argparse --models option
MODEL = "meta-llama/Llama-3.2-3b-Instruct"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# todo move these logics to a separate module
def extract_answer(text: str) -> str:
    """
    Parse model output to find predicted A/B/C/D from patterns like [[A]] or the last letter we see.
    """
    # Quick bracket check
    if "[A]" in text:
        return "A"
    elif "[B]" in text:
        return "B"
    elif "[C]" in text:
        return "C"
    elif "[D]" in text:
        return "D"
    # fallback search from end
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ["A", "B", "C", "D"]:
            return text[i]
    return "A"  # default


def format_prompt_for_test(record, shuffle_choices=True):
    system_msg = SYSTEM_PROMPT.format(field=record.get("subject"))
    user_msg, letter_map = _build_user_message(record, shuffle=shuffle_choices)
    return system_msg, user_msg, letter_map


# todo by default, extract_answer returns 'A' if nothing is found, which may skew results
def evaluate_model_on_dataset(
        model, tokenizer, subject_records, subject, device="cuda"
):
    """
    Evaluate a *finetuned or pruned* model on a list of leftover test records.
    Uses separate system and user messages for proper chat formatting.
    """
    if len(subject_records) == 0:
        return 0.0

    model.eval()
    correct = 0
    for rec in tqdm(
            subject_records, desc=f"Evaluating on {subject}", unit="record"
    ):
        gold = rec.get("answer", "A")

        system_msg, user_msg, letter_map = format_prompt_for_test(
            rec, shuffle_choices=True
        )

        # Format as chat messages using the tokenizer's template
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # Convert to model input format using the tokenizer's chat template
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Convert to tokens
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        model = model.to(device)

        # Generate
        out = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 4,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            top_p=None,
        )
        # Remove the prompt portion
        gen_part = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        raw_letter = extract_answer(gen_text)
        mapped_letter = letter_map.get(raw_letter, "A")

        if mapped_letter == gold:
            correct += 1

    return correct / len(subject_records)


def setup_environment(seed):
    """Initialize environment settings."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()


def setup_tokenizer(model_name):
    """Initialize and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir="./hf_cache")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return tokenizer


def load_model(model_name):
    """Load model with appropriate settings."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, cache_dir="./hf_cache", low_cpu_mem_usage=True
    )
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_raw_model(model, tokenizer, test_num, seed):
    """Evaluate the raw model on all subtasks."""
    subtask_accs = []
    for subject in SUBJECTS:
        for lang in LANGUAGES:
            _, test_recs = get_mmlu(tokenizer, subject, lang, train_num=0, test_num=test_num, seed=seed)
            acc = evaluate_model_on_dataset(
                model, tokenizer, test_recs, subject, device=DEVICE
            )
            subtask_accs.append(acc)
    return subtask_accs


def evaluate_pruned_model(
        model, tokenizer, subject, lang, test_num, seed
):
    """Evaluate a pruned model on all subtasks."""
    subtask_accs = []
    _, test_recs = get_mmlu(tokenizer, subject, lang, train_num=0, test_num=test_num, seed=seed)
    acc = evaluate_model_on_dataset(
        model, tokenizer, test_recs, subject, device=DEVICE
    )
    subtask_accs.append(acc)
    return subtask_accs


def save_pruned_model(model, benchmark, lang, ratio, model_name):
    """Save the pruned model to cache directory."""
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    save_name = (
        f"{os.path.basename(model_name)}_{benchmark}_{lang}_{int(ratio)}pct"
    )
    save_path = os.path.join(cache_dir, "models--" + save_name.replace("/", "--"))
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    return save_path

def load_pruned_model(model_name, benchmark, lang, ratio, device=DEVICE):
    """Load a pruned model saved with the same naming convention."""
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    load_name = f"{os.path.basename(model_name)}_{benchmark}_{lang}_{int(ratio)}pct"
    load_path = os.path.join(cache_dir, "models--" + load_name.replace("/", "--"))
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Pruned model not found at `{load_path}`")
    model = AutoModelForCausalLM.from_pretrained(
        load_path, dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    return model, load_path


def save_results(results_rows, header=None):
    """Save results to CSV file."""
    header = header or ["model_name", "sparsity"] + SUBJECTS
    with open(CSV_NAME, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in results_rows:
            writer.writerow(row)


def prepare_calibration(tokenizer, subject, lang, train_num, seed):
    """Prepare calibration data for pruning."""
    trainloader, _ = get_mmlu(tokenizer, subject, lang, train_num=train_num, test_num=0, seed=seed)
    max_cal_len = max(inp.shape[1] for inp, _, _ in trainloader)
    return trainloader, max_cal_len


def prune() -> None:
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

    # raw_accs = evaluate_raw_model(raw_model, tokenizer, args.test_num, args.seed)
    # results_rows.append(
    #     ["raw", model_name, "None", "0%"] + [f"{acc:.4f}" for acc in raw_accs]
    # )

    # Cleanup raw model
    raw_model.cpu()
    del raw_model
    torch.cuda.empty_cache()

    # Pruned model evaluation
    print(f"\n=== Evaluating PRUNED models for: {MODEL} ===")
    # for subject in SUBJECTS:
    #     for lang in LANGUAGES:
    #         for ratio in args.sparsity_ratios:
    #             print(
    #                 f"\n=== Pruning on subject '{subject}' at {ratio}% sparsity ==="
    #             )
    #
    #             # Initialize new model and tokenizer
    #             base_model = load_model(MODEL)
    #             tokenizer = setup_tokenizer(MODEL)
    #
    #             # Prepare calibration data
    #             trainloader, max_cal_len = prepare_calibration(tokenizer, subject, lang, args.train_num, args.seed)
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
    #                 subject,
    #                 lang,
    #                 args.test_num,
    #                 args.seed,
    #             )
    #
    #             # Save results with new format
    #             results_rows.append(
    #                 ["pruned", model_name, subject, f"{ratio}%"]
    #                 + [f"{acc:.4f}" for acc in subtask_accs]
    #             )
    #
    #             # Save model
    #             save_path = save_pruned_model(
    #                 base_model, subject, lang, ratio, MODEL
    #             )
    #             print(f"Saved pruned model to {save_path}")
    #
    #             # Cleanup
    #             base_model.cpu()
    #             del base_model
    #             torch.cuda.empty_cache()

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
            # benchmark_results = evaluate_model_on_dataset(base_model, tokenizer, benchmark_data, benchmark)

            # Save results with new format
            # results_rows.append(
            #     ["pruned", model_name, benchmark, f"{ratio}%"]
            #     + [f"{metric:.4f}" for metric in benchmark_results]
            # )

            # Save model
            save_path = save_pruned_model(
                base_model, benchmark, "_", ratio, MODEL
            )
            print(f"Saved pruned model to {save_path}")

            # Cleanup
            base_model.cpu()
            del base_model
            torch.cuda.empty_cache()

    # Update save_results call with new header
    header = ["type", "name", "pruned_on", "sparsity"] + SUBJECTS + list(LINGUISTIC_BENCHMARKS.keys())
    save_results(results_rows, header=header)
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
            "benchmark",
            "language",
            "sparsity_ratio",
            "subject",
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
    prune()
