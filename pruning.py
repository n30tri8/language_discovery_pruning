import os
import torch
from transformers import AutoModelForCausalLM
from utils import DEVICE
from submodules.SparseLLM.datautils import get_mmlu

def prepare_calibration(tokenizer, subject, lang, train_num, seed):
    """Prepare calibration data for pruning."""
    trainloader, _ = get_mmlu(tokenizer, subject, lang, train_num=train_num, test_num=0, seed=seed)
    max_cal_len = max(inp.shape[1] for inp, _, _ in trainloader)
    return trainloader, max_cal_len

def save_pruned_model(model, benchmark, lang, ratio, model_name):
    """Save the pruned model to cache directory."""
    cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
    save_name = f"{os.path.basename(model_name)}_{benchmark}_{lang}_{int(ratio)}pct"
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
