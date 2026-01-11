import csv
import json
import os
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from submodules.SparseLLM.mmlu_prompt_templates import MMMLU_PROMPT
from submodules.SparseLLM.prompt_templates import SELECTED_GLUE_TASKS
from submodules.SparseLLM.xglue_loader import load_xnli_test, load_pawsx_test, load_pawsx_italian
from submodules.SparseLLM.xglue_prompt_templates import SELECTED_XGLUE_TASKS, SELECTED_ITALIAN_TASKS, \
    SELECTED_ARABIC_TASKS, SELECTED_HINDI_TASKS


def _shuffle_options(options, letter_map):
    """Helper function to shuffle options while maintaining answer mapping"""
    items = list(zip(options, letter_map.keys()))
    random.shuffle(items)
    shuffled_options, shuffled_keys = zip(*items)
    new_letter_map = {old: new for old, new in zip(letter_map.keys(), shuffled_keys)}
    return shuffled_options, new_letter_map


def _build_user_message(record, lang, shuffle=False):
    # Build the user part for both calibration and testing
    options = [
        record["A"],
        record["B"],
        record["C"],
        record["D"],
    ]
    letter_map = {"A": "A", "B": "B", "C": "C", "D": "D"}

    if shuffle:
        options, letter_map = _shuffle_options(options, letter_map)

    replacement = {"question": record['question'], "options": options}
    user_msg = MMMLU_PROMPT[lang]['user_template'](replacement)

    return user_msg, letter_map


def _build_calibration_prompt(record, tokenizer, lang):
    user_msg, _ = _build_user_message(record, 'en', shuffle=False)  # no shuffle for calibration
    answer = record.get("answer")
    assistant_msg = f"[[{answer}]]"
    replacement = {"field": record.get("subject")}
    system_prompt = MMMLU_PROMPT[lang]['system_template'](replacement)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    text_block = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text_block


def _load_benchmark_data(benchmark_data_dir, subject, lang):
    # Construct the file path based on lang
    file_name = f"{lang}.csv"
    file_path = os.path.join(benchmark_data_dir, file_name)

    data_entries = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            # Assuming columns: id, question, A, B, C, D, answer, subject
            if len(row) < 7:
                continue  # skip malformed rows
            if row[7].strip().lower() == subject.strip().lower():
                entry = {
                    "id": row[0],
                    "question": row[1],
                    "A": row[2],
                    "B": row[3],
                    "C": row[4],
                    "D": row[5],
                    "answer": row[6],
                    "subject": row[7] if len(row) > 7 else subject
                }
                data_entries.append(entry)
    return data_entries


def get_mmlu(benchmark_data_dir, subject, lang, test_num=50):
    records = _load_benchmark_data(benchmark_data_dir, subject, lang)

    return records[:test_num]


# LINGUISTIC BENCHMARKS
def _build_prompts(data, sys, user, assistant):
    """
    Generic builder for chat-style message blocks.

    Args:
        data: iterable dataset (e.g., a HuggingFace Dataset split).
        sys: callable taking a record and returning the system message content.
        user: callable taking a record and returning the user message content.
        assistant: callable taking a record and returning the assistant message content.

    Returns:
        list: List of message lists (each a list of dicts with 'role' and 'content').
    """
    prompts = []

    for record in data:
        messages = [
            {"role": "system", "content": sys(record)},
            {"role": "user", "content": user(record)},
            {"role": "assistant", "content": assistant(record)},
        ]
        prompts.append(messages)

    return prompts


def _tokenize_and_pad(prompts, tokenizer):
    """Tokenize chat prompts and pad to max length like the GLUE loader."""
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    max_len = encoded["input_ids"].shape[1]
    # rearrange to fit into old usage pattern, add a batch dimension of size 1
    rearranged = [(ids.unsqueeze(0), attn.unsqueeze(0)) for ids, attn in zip(encoded["input_ids"], encoded["attention_mask"])]

    return rearranged, max_len


# ENGLISH
def _load_glue_data(task_name, split='train', sample_size=None):
    """
    Load GLUE dataset for the specified task.

    Args:
        task_name (str): The name of the GLUE task (e.g., 'sst2', 'mnli').
        sample_size (int, optional): Number of samples to keep from the train split.
                                     If None, the full train split is returned.

    Returns:
        Dataset: The train split (possibly truncated to `sample_size`).
    """
    dataset = load_dataset("nyu-mll/glue", task_name)
    train = dataset[split]
    if sample_size is None:
        return train
    # Guard against sample_size larger than available examples
    sample_size = min(int(sample_size), len(train))
    return train.select(range(sample_size))


def get_glue(tokenizer):
    """
    Prepare GLUE data for benchmarking, with optional filtering of subsections.

    Args:
        tokenizer: The tokenizer to preprocess the data.
    Returns:
        list: A list of tokenized inputs and labels.
    """

    selected_glue_datasets = {
        task: _load_glue_data(task, sample_size=SELECTED_GLUE_TASKS[task]["sample_size"]) for task in
        SELECTED_GLUE_TASKS.keys()
    }

    for task, dataset in selected_glue_datasets.items():
        # Use the new generic prompt builder by passing the three template callables
        tpl = SELECTED_GLUE_TASKS[task]
        selected_glue_datasets[task] = _build_prompts(
            dataset,
            tpl["system_template"],
            tpl["user_template"],
            tpl["assistant_template"],
        )
        selected_glue_datasets[task] = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ) for messages in selected_glue_datasets[task]
        ]
        selected_glue_datasets[task] = [
            tokenizer(txt, return_tensors="pt", add_special_tokens=False)
            for txt in selected_glue_datasets[task]
        ]

    # todo use _tokenize_and_pad
    raise NotImplementedError
    all_samples = []
    for task in selected_glue_datasets:
        all_samples.extend(selected_glue_datasets[task])
    max_len = max((enc["input_ids"].shape[1] for enc in all_samples), default=0)

    train_loader = []
    for enc in all_samples:
        length = enc["input_ids"].shape[1]
        pad_needed = max_len - length

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        if pad_needed > 0:
            pad_ids = torch.full(
                (1, pad_needed), tokenizer.pad_token_id, dtype=torch.long
            )
            pad_mask = torch.zeros((1, pad_needed), dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        train_loader.append((input_ids, attention_mask))

    return train_loader, max_len


# XGLUE
def _load_xglue_for_calibration(dataset_base_dir, lang) -> Dict[str, List]:
    tasks = {}

    # selectable tasks
    tasks["xnli"] = _build_prompts(
        load_xnli_test(dataset_base_dir, lang, SELECTED_XGLUE_TASKS["xnli"]["sample_size"]),
        SELECTED_XGLUE_TASKS["xnli"][lang]["system_template"],
        SELECTED_XGLUE_TASKS["xnli"][lang]["user_template"],
        SELECTED_XGLUE_TASKS["xnli"][lang]["assistant_template"]
    )

    tasks["pawsx"] = _build_prompts(
        load_pawsx_test(dataset_base_dir, lang, SELECTED_XGLUE_TASKS["pawsx"]["sample_size"]),
        SELECTED_XGLUE_TASKS["pawsx"][lang]["system_template"],
        SELECTED_XGLUE_TASKS["pawsx"][lang]["user_template"],
        SELECTED_XGLUE_TASKS["pawsx"][lang]["assistant_template"]
    )

    return tasks


def get_xglue(tokenizer, base_dir, lang):
    """
    Prepare XGLUE test splits for Wanda calibration, matching get_glue() structure.

    Args:
        tokenizer: tokenizer object
        base_dir (str): path to xglue_full_dataset/
        lang (str): language code (default "de")

    Returns:
        train_loader (list): list of (input_ids, targets, attention_mask)
        max_cal_len (int): max token length observed
    """
    # 1) load the structured chat messages for each task
    selected = _load_xglue_for_calibration(base_dir, lang=lang)

    # 2) Convert system/user/assistant messages to raw chat strings
    for task in selected:
        selected[task] = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in selected[task]
        ]

    # 3) Tokenize + pad
    all_prompts = []
    for task, entries in selected.items():
        all_prompts.extend(entries)

    train_loader, max_cal_len = _tokenize_and_pad(all_prompts, tokenizer)
    return train_loader, max_cal_len


# ITALIAN
def load_uinauil_textualentailmen(base_dir, split, sample_size):
    if split == "dev":
        file = "dev.json"
    elif split == "test":
        file = "test.json"
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'dev' or 'test'.")
    file_path = os.path.join(base_dir, "uinauil-texualentailment", file)

    with open(file_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file {file_path}: {e}")

    selected_size = min(sample_size, len(data))

    return data[:selected_size]


def _load_italian_tasks_for_calibration(benchmark_base_dir) -> Dict[str, List]:
    tasks = {}

    # selectable tasks
    tasks["uinauil-textualentailment"] = _build_prompts(
        load_uinauil_textualentailmen(benchmark_base_dir, split='dev',
                                      sample_size=SELECTED_ITALIAN_TASKS["uinauil-textualentailment"]["sample_size"]),
        SELECTED_ITALIAN_TASKS["uinauil-textualentailment"]["system_template"],
        SELECTED_ITALIAN_TASKS["uinauil-textualentailment"]["user_template"],
        SELECTED_ITALIAN_TASKS["uinauil-textualentailment"]["assistant_template"]
    )

    tasks["pawsx-translated"] = _build_prompts(
        load_pawsx_italian(benchmark_base_dir, SELECTED_ITALIAN_TASKS["pawsx-translated"]["sample_size"], split="dev"),
        SELECTED_ITALIAN_TASKS["pawsx-translated"]["system_template"],
        SELECTED_ITALIAN_TASKS["pawsx-translated"]["user_template"],
        SELECTED_ITALIAN_TASKS["pawsx-translated"]["assistant_template"]
    )

    return tasks


def get_italian_calib(tokenizer, base_dir):
    """
    Prepare italian test splits for Wanda calibration, matching get_glue() structure.

    Args:
        tokenizer: tokenizer object
        base_dir (str): path to benchmark_data/

    Returns:
        train_loader (list): list of (input_ids, targets, attention_mask)
        max_cal_len (int): max token length observed
    """
    # 1) load the structured chat messages for each task
    selected = _load_italian_tasks_for_calibration(base_dir)

    # 2) Convert system/user/assistant messages to raw chat strings
    for task in selected:
        selected[task] = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in selected[task]
        ]

    # 3) Tokenize + pad
    all_prompts = []
    for task, entries in selected.items():
        all_prompts.extend(entries)

    train_loader, max_cal_len = _tokenize_and_pad(all_prompts, tokenizer)
    return train_loader, max_cal_len


# ARABIC
def load_paraphrase_arabic(base_dir, split, sample_size):
    if split == "dev":
        file = "dev.csv"
    elif split == "test":
        file = "test.csv"
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'dev' or 'test'.")
    file_path = os.path.join(base_dir, "Arabic-Paraphrasing-Benchmark", file)

    data = []
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)  # assumes header: sentence1,sentence2,label
            for row in reader:
                if not row:
                    continue  # ignore fully empty rows
                # We assume the dataset is clean and has these keys
                record = {
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                    "label": row["label"],  # kept as string, e.g. "0"/"1"
                }
                data.append(record)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}")

    selected_size = min(int(sample_size), len(data))

    return data[:selected_size]


def _load_arabic_tasks_for_calibration(benchmark_base_dir) -> Dict[str, List]:
    tasks = {}

    # selectable tasks
    xnli_base_dir = os.path.join(benchmark_base_dir, "xglue_dataset")
    tasks["xnli"] = _build_prompts(
        load_xnli_test(xnli_base_dir, lang='ar', split='dev',
                       sample_size=SELECTED_ARABIC_TASKS["xnli"]["sample_size"]),
        SELECTED_ARABIC_TASKS["xnli"]["system_template"],
        SELECTED_ARABIC_TASKS["xnli"]["user_template"],
        SELECTED_ARABIC_TASKS["xnli"]["assistant_template"]
    )

    tasks["paraphrase"] = _build_prompts(
        load_paraphrase_arabic(benchmark_base_dir, sample_size=SELECTED_ARABIC_TASKS["paraphrase"]["sample_size"],
                               split="dev"),
        SELECTED_ARABIC_TASKS["paraphrase"]["system_template"],
        SELECTED_ARABIC_TASKS["paraphrase"]["user_template"],
        SELECTED_ARABIC_TASKS["paraphrase"]["assistant_template"]
    )

    return tasks


def get_arabic_calib(tokenizer, base_dir):
    """
    Prepare arabic test splits for Wanda calibration, matching get_glue() structure.

    Args:
        tokenizer: tokenizer object
        base_dir (str): path to benchmark_data/

    Returns:
        train_loader (list): list of (input_ids, targets, attention_mask)
        max_cal_len (int): max token length observed
    """
    # 1) load the structured chat messages for each task
    selected = _load_arabic_tasks_for_calibration(base_dir)

    # 2) Convert system/user/assistant messages to raw chat strings
    for task in selected:
        selected[task] = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in selected[task]
        ]

    # 3) Tokenize + pad
    all_prompts = []
    for task, entries in selected.items():
        all_prompts.extend(entries)

    train_loader, max_cal_len = _tokenize_and_pad(all_prompts, tokenizer)
    return train_loader, max_cal_len


# Hindi
def load_paraphrase_hindi(base_dir, split, sample_size):
    if split == "dev":
        file = "dev.csv"
    elif split == "test":
        file = "test.csv"
    else:
        raise ValueError(f"Unsupported split: {split}. Expected 'dev' or 'test'.")
    file_path = os.path.join(base_dir, "IndicParaphrase", file)

    data = []
    try:
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)  # assumes header: sentence1,sentence2,label
            for row in reader:
                if not row:
                    continue  # ignore fully empty rows
                # We assume the dataset is clean and has these keys
                record = {
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                    "label": row["label"],  # kept as string, e.g. "0"/"1"
                }
                data.append(record)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file {file_path}: {e}")

    selected_size = min(int(sample_size), len(data))

    return data[:selected_size]


def _load_hindi_tasks_for_calibration(benchmark_base_dir) -> Dict[str, List]:
    tasks = {}

    # selectable tasks
    xnli_base_dir = os.path.join(benchmark_base_dir, "xglue_dataset")
    tasks["xnli"] = _build_prompts(
        load_xnli_test(xnli_base_dir, lang='hi', split='dev',
                       sample_size=SELECTED_HINDI_TASKS["xnli"]["sample_size"]),
        SELECTED_HINDI_TASKS["xnli"]["system_template"],
        SELECTED_HINDI_TASKS["xnli"]["user_template"],
        SELECTED_HINDI_TASKS["xnli"]["assistant_template"]
    )

    tasks["paraphrase"] = _build_prompts(
        load_paraphrase_hindi(benchmark_base_dir, sample_size=SELECTED_HINDI_TASKS["paraphrase"]["sample_size"],
                              split="dev"),
        SELECTED_HINDI_TASKS["paraphrase"]["system_template"],
        SELECTED_HINDI_TASKS["paraphrase"]["user_template"],
        SELECTED_HINDI_TASKS["paraphrase"]["assistant_template"]
    )

    return tasks


def get_hindi_calib(tokenizer, base_dir):
    """
    Prepare hindi test splits for Wanda calibration, matching get_glue() structure.

    Args:
        tokenizer: tokenizer object
        base_dir (str): path to benchmark_data/

    Returns:
        train_loader (list): list of (input_ids, targets, attention_mask)
        max_cal_len (int): max token length observed
    """
    # 1) load the structured chat messages for each task
    selected = _load_hindi_tasks_for_calibration(base_dir)

    # 2) Convert system/user/assistant messages to raw chat strings
    for task in selected:
        selected[task] = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            for messages in selected[task]
        ]

    # 3) Tokenize + pad
    all_prompts = []
    for task, entries in selected.items():
        all_prompts.extend(entries)

    train_loader, max_cal_len = _tokenize_and_pad(all_prompts, tokenizer)
    return train_loader, max_cal_len


# test cases
def test_get_mmlu():
    subject, lang = "management", "EN"

    benchmark_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_data", "mmlu"))
    test_records = get_mmlu(benchmark_data_dir, subject, lang, test_num=2)

    print("Test records samples:")
    for i, record in enumerate(test_records):
        print(f"Test Record {i}: {record}")


def test_get_glue():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_fast=False,
                                              cache_dir="./hf_cache")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    combined_data = get_glue(tokenizer)

    print("Combined GLUE Data Samples:")
    for i, (inputs, label) in enumerate(combined_data[:5]):  # Print first 5 samples
        print(f"Sample {i}:")
        print("Inputs:", inputs)
        print("Label:", label)
        print()


def test_xglue():
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct", use_fast=False, cache_dir=os.path.expanduser("~/.cache/huggingface/hub")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    base_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_data", "xglue_dataset")
    )
    train_loader, max_cal_len = get_xglue(tokenizer, base_dir, lang="de")

    print(f"XGLUE train samples: {len(train_loader)}, max_cal_len: {max_cal_len}")
    for i, (input_ids, targets, attention_mask) in enumerate(train_loader[:3]):
        print(f"Sample {i}: targets={targets}")


if __name__ == "__main__":
    test_xglue()
