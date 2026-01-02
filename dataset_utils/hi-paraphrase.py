import json
import random

import pandas as pd

RANDOM_SEED = 42
TOTAL_SAMPLES = 200
POSITIVE_RATIO = 0.5

INPUT_FILE = "D:\\repos\language_discovery_pruning\\benchmark_data\IndicParaphrase\\test_hi.jsonl"  # path to your dev.jsonl
OUTPUT_FILE = "D:\\repos\language_discovery_pruning\\benchmark_data\IndicParaphrase\\test.csv"  # output file

random.seed(RANDOM_SEED)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_dataset(records):
    """
    Creates a dataset with columns:
    original_id | sentence1 | sentence2 | label
    """

    num_positive = int(TOTAL_SAMPLES * POSITIVE_RATIO)
    num_negative = TOTAL_SAMPLES - num_positive

    # ---------
    # POSITIVE PAIRS
    # ---------
    positive_pairs = []

    for r in records:
        s1 = r["input"]
        orig_id = r["id"]

        # input → target
        positive_pairs.append({
            "original_id": orig_id,
            "sentence1": s1,
            "sentence2": r["target"],
            "label": 1
        })

        # input → references
        # for ref in r.get("references", []):
        #     positive_pairs.append({
        #         "original_id": orig_id,
        #         "sentence1": s1,
        #         "sentence2": ref,
        #         "label": 1
        #     })

    random.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:num_positive]

    # ---------
    # NEGATIVE PAIRS
    # Sample across examples
    # ---------
    negatives = []

    while len(negatives) < num_negative:
        i, j = random.sample(range(len(records)), 2)

        r1 = records[i]
        r2 = records[j]

        s1 = r1["input"]
        s2 = r2["target"]

        # Avoid accidental positives
        if s2 in r1.get("references", []):
            continue

        negatives.append({
            "original_id": r1["id"],  # keep id of sentence1 source
            "sentence1": s1,
            "sentence2": s2,
            "label": 0
        })

    # ---------
    # COMBINE + SHUFFLE
    # ---------
    dataset = positive_pairs + negatives
    random.shuffle(dataset)

    return dataset


def main():
    records = load_jsonl(INPUT_FILE)
    dataset = prepare_dataset(records)

    df = pd.DataFrame(dataset)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(df)} samples to {OUTPUT_FILE}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
