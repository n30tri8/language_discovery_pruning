###################################
# Directory-driven file loading   #
###################################
import csv
import os
from submodules.SparseLLM.xglue_prompt_templates import SELECTED_XGLUE_TASKS


def _safe_get(d, key):
    return d[key] if key in d else ""


def load_xnli_test(base_dir: str, lang: str, sample_size: int):
    file = os.path.join(base_dir, "XNLI", f"{lang}.test")
    data = []
    with open(file, encoding="utf-8") as f:
        processed = 0
        for line in f:
            parts = line.strip().split("\t", maxsplit=2)
            if len(parts) != 3:
                continue
            data.append({
                "premise": parts[0].strip(),
                "hypothesis": parts[1].strip(),
                "label": parts[2].strip()
            })
            processed += 1
            if processed >= sample_size:
                break
    return data


def load_pawsx_test(base_dir: str, lang: str, sample_size: int):
    file = os.path.join(base_dir, "PAWSX", lang, "test_2k.tsv")
    data = []
    with open(file, encoding="utf-8") as f:
        processed = 0
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            data.append({
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"],
                "label_text": row["label"]
            })
            processed += 1
            if processed >= sample_size:
                break
    return data


def load_nc_test(base_dir: str, lang: str, sample_size: int):
    file = os.path.join(base_dir, "NC", f"xglue.nc.{lang}.test")
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 3:
                data.append({
                    "title": parts[1],
                    "body": parts[2],
                    "label": "<unknown>"
                })
    return data[:sample_size] if sample_size else data


def load_qam_test(base_dir: str, lang: str, sample_size: int):
    file = os.path.join(base_dir, "QAM", f"xglue.qam.{lang}.test")
    data = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 2:
                data.append({
                    "question": parts[0],
                    "passage": parts[1],
                    "label_text": "<unknown>"
                })
    return data[:sample_size] if sample_size else data
