import torch
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


def format_system_prompt(subject, lang):
    replacement = {"field": subject}
    system_msg = MMMLU_PROMPT[lang]['system_template'](replacement)
    return system_msg

def format_user_prompt(record, lang, shuffle_choices=True):
    user_msg, letter_map = _build_user_message(record, lang, shuffle=shuffle_choices)
    return user_msg, letter_map


import time


@torch.inference_mode()
def evaluate_model_on_dataset(model, tokenizer, subject_records, subject, lang, device="cuda", batch_size=10):
    """
    Evaluate a *finetuned or pruned* model on a list of leftover test records.
    """
    if len(subject_records) == 0:
        return 0.0

    model = model.to(device)
    correct = 0

    batches = [
        subject_records[i:i + batch_size]
        for i in range(0, len(subject_records), batch_size)
    ]

    # Pre-cache the system prompt as they are the same for all records
    system_msg = format_system_prompt(subject, lang)

    for batch in tqdm(batches, desc=f"Evaluating on {subject}", unit="batch"):
        user_msgs, letter_maps, correct_answers = [], [], []

        # Preprocess batch
        for rec in batch:
            user_msg, letter_map = format_user_prompt(rec, lang, shuffle_choices=True)
            user_msgs.append(user_msg)
            letter_maps.append(letter_map)
            correct_answers.append(rec.get("answer"))

        # Tokenize batch
        messages = [
            [{"role": "system", "content": system_msg}, {"role": "user", "content": usr_msg}]
            for usr_msg in user_msgs
        ]
        chat_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                      messages]
        inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        t0 = time.time()
        # Generate outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            top_p=None,
        )
        t_gen = time.time()

        print(
            f"generate={t_gen - t0:.4f}s"
        )

        # Decode and evaluate
        input_length = inputs["input_ids"].shape[1]
        for i, out in enumerate(outputs):
            gen_part = out[input_length:]
            gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)
            model_extracted_answer = extract_answer(gen_text)
            mapped_answer = letter_maps[i].get(model_extracted_answer)
            if mapped_answer == correct_answers[i]:
                correct += 1

    return correct / len(subject_records)


def evaluate_model(model, tokenizer, benchmark_data_dir, subject, lang, test_num):
    """Evaluate a pruned model on a subject*lang ."""
    _, test_recs = get_mmlu(tokenizer, benchmark_data_dir, subject, lang, train_num=0, test_num=test_num)
    acc = evaluate_model_on_dataset(model, tokenizer, test_recs, subject, lang, device=DEVICE)
    return acc
