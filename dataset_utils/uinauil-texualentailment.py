import json

import xmltodict

with open(r"D:\repos\language_discovery_pruning\benchmark_data\uinauil-texualentailment\dev.xml", "r",
          encoding="utf-8") as f:
    xml = f.read()

data = xmltodict.parse(xml)

converted = []
# handle case where there's a single <pair> (dict) or multiple (list)
pairs = data.get("entailment-corpus", {}).get("pair", [])
if isinstance(pairs, dict):
    pairs = [pairs]

for pair in pairs:
    label_text = pair.get("@entailment")
    if label_text == "YES":
        label = "1"
    else:
        label = "0"
    new_pair = {
        "premise": pair.get("t"),
        "hypothesis": pair.get("h"),
        "label": label,
        "id": pair.get("@id")
    }
    converted.append(new_pair)

# save converted to a .json file
json_path = r"D:\repos\language_discovery_pruning\benchmark_data\uinauil-texualentailment\dev.json"
with open(json_path, "w", encoding="utf-8") as out:
    json.dump(converted, out, ensure_ascii=False, indent=2)

# converting test data
with open(r"D:\repos\language_discovery_pruning\benchmark_data\uinauil-texualentailment\test_gold.xml", "r",
          encoding="utf-8") as f:
    xml = f.read()

data = xmltodict.parse(xml)

converted = []
# handle case where there's a single <pair> (dict) or multiple (list)
pairs = data.get("entailment-corpus", {}).get("pair", [])
if isinstance(pairs, dict):
    pairs = [pairs]

for pair in pairs:
    label_text = pair.get("@entailment")
    if label_text == "YES":
        label = "1"
    else:
        label = "0"
    new_pair = {
        "premise": pair.get("t"),
        "hypothesis": pair.get("h"),
        "label": label,
        "id": pair.get("@id")
    }
    converted.append(new_pair)

# save converted to a .json file
json_path = r"D:\repos\language_discovery_pruning\benchmark_data\uinauil-texualentailment\test.json"
with open(json_path, "w", encoding="utf-8") as out:
    json.dump(converted, out, ensure_ascii=False, indent=2)
