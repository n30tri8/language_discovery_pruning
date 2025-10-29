from tqdm import tqdm
from utils import DEVICE
from submodules.SparseLLM.datautils import _build_user_message, SYSTEM_PROMPT
from submodules.SparseLLM.datautils import get_mmlu

def extract_answer(text: str) -> str:
    """
    Parse model output to find predicted A/B/C/D from patterns like [[A]] or the last letter we see.
    """
    if "[A]" in text:
        return "A"
    elif "[B]" in text:
        return "B"
    elif "[C]" in text:
        return "C"
    elif "[D]" in text:
        return "D"
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ["A", "B", "C", "D"]:
            return text[i]
    return "A"  # default

def format_prompt_for_test(record, shuffle_choices=True):
    system_msg = SYSTEM_PROMPT.format(field=record.get("subject"))
    user_msg, letter_map = _build_user_message(record, shuffle=shuffle_choices)
    return system_msg, user_msg, letter_map

# todo by default, extract_answer returns 'A' if nothing is found, which may skew results
def evaluate_model_on_dataset(model, tokenizer, subject_records, subject, device="cuda"):
    """
    Evaluate a *finetuned or pruned* model on a list of leftover test records.
    """
    if len(subject_records) == 0:
        return 0.0

    model.eval()
    correct = 0
    for rec in tqdm(subject_records, desc=f"Evaluating on {subject}", unit="record"):
        gold = rec.get("answer", "A")
        system_msg, user_msg, letter_map = format_prompt_for_test(rec, shuffle_choices=True)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        model = model.to(device)
        out = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 4,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            top_p=None,
        )
        gen_part = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        raw_letter = extract_answer(gen_text)
        mapped_letter = letter_map.get(raw_letter, "A")
        if mapped_letter == gold:
            correct += 1
    return correct / len(subject_records)

def evaluate_raw_model_on_mmlu(model, tokenizer, benchmark_data_dir, test_num, seed, subjects, languages):
    """Evaluate the raw model on all subtasks."""
    subtask_accs = []
    for subject in subjects:
        for lang in languages:
            _, test_recs = get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=0, test_num=test_num,
                                    seed=seed)
            acc = evaluate_model_on_dataset(model, tokenizer, test_recs, subject, device=DEVICE)
            subtask_accs.append(acc)
    return subtask_accs

# todo refactor, similar to another function above
def evaluate_pruned_model(model, tokenizer, benchmark_data_dir, subject, lang, test_num, seed):
    """Evaluate a pruned model on all subtasks."""
    subtask_accs = []
    _, test_recs = get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=0, test_num=test_num, seed=seed)
    acc = evaluate_model_on_dataset(
        model, tokenizer, test_recs, subject, device=DEVICE
    )
    subtask_accs.append(acc)
    return subtask_accs