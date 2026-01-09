import torch
from tqdm import tqdm

from utils import DEVICE


class EvalSpec:
    def __init__(self, selected_tasks, benchmark_name, lang):
        self.tasks = None
        self.benchmark_name: str = benchmark_name
        self.selected_tasks = selected_tasks
        self.tasks = self.selected_tasks.keys()
        self.lang = lang

    def load_eval_data(self, **kwargs):
        pass

    def build_system_prompt(self, task, record):
        sys_prompt = self.selected_tasks[task]["system_template"](record)

        return sys_prompt

    def build_user_prompt(self, task, record):
        user_prompt = self.selected_tasks[task]["user_template"](record)

        return user_prompt

    def correct_answer(self, record, **kwargs):
        answer = str(record["label"])

        return answer

    def extract_answer(self, generated_text, **kwargs):
        pass


@torch.inference_mode()
def evaluate_on_linguistic(model, tokenizer, evaluation_spec: EvalSpec, batch_size=50, device=DEVICE):
    task_accuracies = {}

    for task in evaluation_spec.tasks:
        records = evaluation_spec.load_eval_data(task=task)
        if len(records) == 0:
            task_accuracies[task] = 0.0
            continue

        correct = 0

        batches = [
            [records[j] for j in range(i, min(i + batch_size, len(records)))]
            for i in range(0, len(records), batch_size)
        ]

        # Pre-cache the system prompt as they are the same for all records
        system_msg = evaluation_spec.build_system_prompt(task=task, record=records[0])

        for batch in batches:
            # Preprocess batch
            user_msgs, correct_answers = [], []
            for rec in batch:
                user_msg = evaluation_spec.build_user_prompt(task=task, record=rec)
                user_msgs.append(user_msg)
                crct_ans = evaluation_spec.correct_answer(record=rec)
                correct_answers.append(crct_ans)

            # Tokenize batch
            messages = [
                [{"role": "system", "content": system_msg}, {"role": "user", "content": usr_msg}]
                for usr_msg in user_msgs
            ]
            chat_texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in
                          messages]
            inputs = tokenizer(chat_texts, return_tensors="pt", padding=True, truncation=True)

            # Generate outputs
            outputs = model.generate(
                **inputs,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                top_p=None,
            )

            # Decode and evaluate
            input_length = inputs["input_ids"].shape[1]
            for idx in range(len(batch)):
                out = outputs[idx,:]
                gen_part = out[input_length:]
                gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)
                model_extracted_answer = evaluation_spec.extract_answer(gen_text, task=task)
                if model_extracted_answer == correct_answers[idx]:
                    correct += 1

        task_accuracy = correct / len(records)
        task_accuracies[task] = task_accuracy

    return task_accuracies
