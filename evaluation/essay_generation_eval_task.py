import gc

import torch

from benchmark_loader.essay_generation_prompt_templates import philosophical_essay_generation_template


def essay_generation_test(model, tokenizer, lang):
    user_message = philosophical_essay_generation_template[lang]
    messages = [
        [{"role": "system", "content": ""}, {"role": "user", "content": user_message}]
    ]
    chat_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                  messages]
    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=2000,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        top_p=None,
    ).cpu()

    input_length = inputs["input_ids"].shape[1]
    # free GPU memory
    del inputs
    gc.collect()
    torch.cuda.empty_cache()

    gen_part = outputs[0, input_length:] # batch size=1
    gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)

    return gen_text


