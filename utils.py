import csv
import os
import random
import threading

import numpy as np
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Default cache dir; can be overridden by setup_environment
RAW_MODEL_DIR = "raw_model"


def setup_environment(seed, raw_model_dir):
    """Initialize environment settings and (optionally) set the HF cache dir.

    Args:
        seed (int): random seed to set for numpy, random and torch.
        raw_model_dir (str|None): path to use as the raw model dir. If provided,
            this will override the module default HF_CACHE_DIR used by tokenizer/model loaders.
    """
    global RAW_MODEL_DIR

    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()

    # login to huggingface hub if token present
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    RAW_MODEL_DIR = raw_model_dir


def setup_tokenizer(model_name):
    """Initialize and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=RAW_MODEL_DIR,
                                              padding_side='left'  # for decoder-only models
                                              )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  # inference-safe

    return tokenizer


def load_raw_model(model_name):
    """Load model with appropriate settings."""
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=RAW_MODEL_DIR,
                                                 dtype=torch.float16,
                                                 # float16, float32 for cpu
                                                 device_map="auto"  # for multi gpu support
                                                 )
    # Ensure deterministic generation
    model.generation_config.do_sample = False
    model.generation_config.top_p = None
    model.generation_config.temperature = None
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


def save_pruned_model_async(model, save_path):
    """Save the pruned model asynchronously to avoid blocking the main thread."""

    def _save(m):
        os.makedirs(save_path, exist_ok=True)
        print(f"[INFO] Saving pruned model to {save_path}...")
        m.save_pretrained(save_path)
        print(f"[INFO] ✅ Model saved: {save_path}")
        # Cleanup
        del m

    # Launch background thread
    thread = threading.Thread(target=_save, args=(model,), daemon=False)
    thread.start()

    return thread  # Return thread if caller wants to join()


def load_pruned_model(load_path, device=DEVICE):
    """Load a pruned model saved with the same naming convention."""
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"Pruned model not found at `{load_path}`")
    model = AutoModelForCausalLM.from_pretrained(
        load_path, dtype=torch.float16, device_map="auto"
    )
    # Ensure deterministic generation
    model.generation_config.do_sample = False
    model.generation_config.top_p = None
    model.generation_config.temperature = None
    model.eval()
    return model, load_path
