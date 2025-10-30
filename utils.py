import csv
import os
import random

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_environment(seed):
    """Initialize environment settings."""
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    login(token=os.getenv("HF_TOKEN"))
    # hf_cache_dir =


def setup_tokenizer(model_name):
    """Initialize and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir="./hf_cache")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    return tokenizer


def load_raw_model(model_name):
    """Load model with appropriate settings."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, cache_dir="./hf_cache", low_cpu_mem_usage=True
    )
    model = model.to(DEVICE)
    model.eval()
    return model


def save_results(output_file, results_rows, output_cols, header=None):
    """Save results to CSV file."""
    header = header or ["model_name", "sparsity"] + output_cols
    with open(output_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in results_rows:
            writer.writerow(row)


def model_dir(pruned_model_dir, model_name, benchmark, lang, ratio):
    """Construct directory path for pruned model."""
    save_name = f"{os.path.basename(model_name)}_{benchmark}_{lang}_{int(ratio)}pct"
    save_name = "models--" + save_name.replace("/", "--")
    return os.path.join(pruned_model_dir, save_name)


def save_pruned_model(model, save_path):
    """Save the pruned model to cache directory."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    return save_path


def load_pruned_model(load_path, device=DEVICE):
    """Load a pruned model saved with the same naming convention."""
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Pruned model not found at `{load_path}`")
    model = AutoModelForCausalLM.from_pretrained(
        load_path, dtype=torch.float16, low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()
    return model, load_path
