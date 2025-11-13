from tqdm import tqdm

from submodules.SparseLLM.datautils import _build_user_message
from submodules.SparseLLM.datautils import get_mmlu
from submodules.SparseLLM.mmlu_prompt_templates import MMMLU_PROMPT
from utils import DEVICE


def extract_answer(text: str) -> str:
    """
    Parse model output to find predicted A/B/C/D from patterns like [[A]] or the last letter we see.
    """
    # todo, not suitable for languages that have different alphabet
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
    return " "  # default if nothing found


def format_prompt_for_test(record, lang, shuffle_choices=True):
    replacement = {"field": record.get("subject")}
    system_msg = MMMLU_PROMPT[lang]['system_template'](replacement)
    user_msg, letter_map = _build_user_message(record, lang, shuffle=shuffle_choices)
    return system_msg, user_msg, letter_map


def evaluate_model_on_dataset(model, tokenizer, subject_records, subject, lang, device="cuda"):
    """
    Evaluate a *finetuned or pruned* model on a list of leftover test records.
    """
    if len(subject_records) == 0:
        return 0.0

    model = model.to(device)
    tokenizer = tokenizer.to(device)
    model.eval()

    correct = 0
    for rec in tqdm(subject_records, desc=f"Evaluating on {subject}", unit="record"):
        system_msg, user_msg, letter_map = format_prompt_for_test(rec, lang, shuffle_choices=True)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            top_p=None,
        )
        gen_part = out[0][inputs["input_ids"].shape[1]:]
        gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)
        model_extracted_answer = extract_answer(gen_text)
        mapped_answer = letter_map.get(model_extracted_answer)
        correct_answer = rec.get("answer")
        if mapped_answer == correct_answer:
            correct += 1
    return correct / len(subject_records)


def evaluate_model(model, tokenizer, benchmark_data_dir, subject, lang, test_num):
    """Evaluate a pruned model on a subject*lang ."""
    _, test_recs = get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=0, test_num=test_num)
    acc = evaluate_model_on_dataset(model, tokenizer, test_recs, subject, lang, device=DEVICE)
    return acc
