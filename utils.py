import csv

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def save_results(output_file, results_rows, output_cols, header=None):
    """Save results to CSV file."""
    header = header or ["model_name", "sparsity"] + output_cols
    with open(output_file, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for row in results_rows:
            writer.writerow(row)
