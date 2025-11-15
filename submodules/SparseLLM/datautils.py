import csv
import os
import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from submodules.SparseLLM.mmlu_prompt_templates import MMMLU_PROMPT
from submodules.SparseLLM.prompt_templates import SELECTED_GLUE_TASKS
from submodules.SparseLLM.xglue_loader import load_xnli_test, load_pawsx_test
from submodules.SparseLLM.xglue_prompt_templates import SELECTED_XGLUE_TASKS


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

    replacement = {"question":record['question'], "options":options}
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


def get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=32, test_num=5):
    """
    Prepare the ToMBench calibration data (trainloader) and test data (list of leftover records).

    1) Loads subtask_file from ToMBench.
    2) Splits into 'train_num' for calibration vs. remainder for test.
    3) Builds a "calibration prompt" for each train record, then tokenizes & pads to the max length across them.
       The returned 'trainloader' is a list of (inp, tar) pairs, each shaped [1, seq_len].
       - We do *not* fix a seqlen; we let them pad to the longest sample.
    4) The test set is limited to `test_num` samples (default 5). For each test record, we keep it raw (a dict).
       We'll handle prompting in the evaluation code.

    Returns:
      trainloader: list of (inp, tar) pairs ready for unstructured pruning with e.g. llama_sparsellm
      test_records: list of leftover records (the test data)

    Example usage:
      trainloader, test_recs = get_tom(tokenizer, "False Belief Task.jsonl", 32, 5)
    """
    records = _load_benchmark_data(benchmark_data_dir, subject, lang)

    # 1) Split
    train_records = records[:train_num]
    leftover_records = records[train_num:]

    # 2) Build calibration prompts

    train_prompts = [_build_calibration_prompt(r, tokenizer, lang) for r in train_records]

    # 3) Tokenize each prompt (variable length)
    encoded_list = []
    for txt in train_prompts:
        enc = tokenizer(txt, return_tensors="pt", add_special_tokens=False)
        # shape: [1, length]
        encoded_list.append(enc)

    # Find max length
    max_len = 0
    for enc in encoded_list:
        length = enc["input_ids"].shape[1]
        if length > max_len:
            max_len = length

    # 4) Pad each to max_len and build (inp, tar)
    trainloader = []
    for enc in encoded_list:
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

        # Now create the targets so that we are effectively doing next-token or LM-style supervision.
        tar = input_ids.clone()
        # Standard trick: mask everything except the “shifted by 1”
        tar[:, :-1] = -100

        trainloader.append((input_ids, tar, attention_mask))

    # 5) Limit test set size to test_num (if positive)
    test_records = leftover_records
    if test_num is not None and test_num > 0:
        test_records = test_records[:test_num]

    return trainloader, test_records


def _load_xglue_data(task_name, sample_size=None):
    """
    Load XGLUE dataset for the specified task.

    Uses `load_dataset("microsoft/xglue", task_name)` and returns the train split
    (or the first available split if a train split is not present).

    Args:
        task_name (str): The name of the XGLUE task/config.
        sample_size (int, optional): Number of samples to keep from the train split.
                                     If None, the full train split is returned.

    Returns:
        Dataset: The train split (possibly truncated to `sample_size`).
    """
    dataset = load_dataset("microsoft/xglue", task_name)
    # Prefer explicit "train" split, otherwise fall back to the first split available
    if "train" in dataset:
        split = dataset["train"]
    else:
        first_split = list(dataset.keys())[0]
        split = dataset[first_split]

    if sample_size is None:
        return split

    sample_size = min(int(sample_size), len(split))
    return split.select(range(sample_size))


def _load_glue_data(task_name, sample_size=None):
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
    train = dataset["train"]
    if sample_size is None:
        return train
    # Guard against sample_size larger than available examples
    sample_size = min(int(sample_size), len(train))
    return train.select(range(sample_size))


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


def get_glue(tokenizer):
    """
    Prepare GLUE data for benchmarking, with optional filtering of subsections.

    Args:
        tokenizer: The tokenizer to preprocess the data.
    Returns:
        list: A list of tokenized inputs and labels.
    """

    selected_glue_datasets = {
        task: _load_glue_data(task, SELECTED_GLUE_TASKS[task]["sample_size"]) for task in SELECTED_GLUE_TASKS.keys()
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

        # Now create the targets so that we are effectively doing next-token or LM-style supervision.
        tar = input_ids.clone()
        # Standard trick: mask everything except the “shifted by 1”
        tar[:, :-1] = -100

        train_loader.append((input_ids, tar, attention_mask))

    max_cal_len = max(inp.shape[1] for inp, _, _ in train_loader)

    return train_loader, max_cal_len


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


def _tokenize_and_pad(prompts, tokenizer):
    """Tokenize chat prompts and pad to max length like the GLUE loader."""
    # First tokenize everything without padding
    encoded = [
        tokenizer(
            txt, return_tensors="pt", add_special_tokens=False
        )
        for txt in prompts
    ]

    # Compute max length
    max_len = max(enc["input_ids"].shape[1] for enc in encoded)

    # Pad all to max length
    processed = []
    for enc in encoded:
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        pad_needed = max_len - input_ids.shape[1]
        if pad_needed > 0:
            pad_ids = torch.full(
                (1, pad_needed), tokenizer.pad_token_id, dtype=torch.long
            )
            pad_mask = torch.zeros((1, pad_needed), dtype=torch.long)

            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        # now create shifted-target (Wanda does not use it but format matches GLUE)
        targets = input_ids.clone()
        targets[:, :-1] = -100  # mask-out all except “next token”

        processed.append((input_ids, targets, attention_mask))

    return processed, max_len


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


def test_get_mmlu():
    subject, lang = "management", "EN"
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", use_fast=False,
                                              cache_dir="./hf_cache")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    benchmark_data_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_data", "mmlu"))
    trainloader, test_records = get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=2, test_num=2)

    print("Trainloader samples:")
    for i, (input_ids, tar, attention_mask) in enumerate(trainloader):
        print(f"Sample {i}:")
        print("input_ids:", input_ids)
        print("tar:", tar)
        print("attention_mask:", attention_mask)
        print()

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
