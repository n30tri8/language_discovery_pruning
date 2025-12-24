from evaluation.common_evaluation import EvalSpec
from submodules.SparseLLM.datautils import _load_glue_data
from submodules.SparseLLM.prompt_templates import SELECTED_GLUE_TASKS


class GlueEvalSpec(EvalSpec):
    def __init__(self):
        super().__init__()
        self.selected_tasks = SELECTED_GLUE_TASKS

        self.tasks = self.selected_tasks.keys()
        self.benchmark_name = "GLUE-EN"

    def load_eval_data(self, task):
        if task == "mnli":
            data = _load_glue_data(task, split='validation_matched', sample_size=self.selected_tasks[task]["test_size"])
        else:
            data = _load_glue_data(task, split='validation', sample_size=self.selected_tasks[task]["test_size"])

        return data

    def build_system_prompt(self, task, record):
        sys_prompt = self.selected_tasks[task]["system_template"](record)

        return sys_prompt

    def build_user_prompt(self, task, record):
        user_prompt = self.selected_tasks[task]["user_template"](record)

        return user_prompt

    def correct_answer(self, task, record):
        answer = str(record["label"])

        return answer

    def extract_answer(self, generated_text, **kwargs):
        """
        Extract the answer from generated text based on task-specific labels.
        Returns the expected label or empty string if not found.
        """
        task = kwargs.get('task')
        if not task:
            return ""

        # Clean the generated text
        text = generated_text.strip()

        # Define valid labels for each task
        valid_labels = {
            "mnli": ["0", "1", "2"],  # 0=entailment, 1=neutral, 2=contradiction
            "cola": ["0", "1"],       # 0=unacceptable, 1=acceptable
            "qqp": ["0", "1"]         # 0=not paraphrases, 1=paraphrases
        }

        task_labels = valid_labels[task]

        # Look for patterns like [0], [1], [2] first
        for label in task_labels:
            if f"[{label}]" in text:
                return label

        # Look for patterns like "0", "1", "2" in quotes
        for label in task_labels:
            if f'"{label}"' in text or f"'{label}'" in text:
                return label

        # Look for the label at the beginning of the text (common pattern)
        if text and text[0] in task_labels:
            return text[0]

        # Look for the last occurrence of any valid label in the text
        for i in range(len(text) - 1, -1, -1):
            if text[i] in task_labels:
                return text[i]

        # Look for words that might indicate the answer
        text_lower = text.lower()

        # For mnli task, look for word-based answers
        if task == "mnli":
            if "entailment" in text_lower or "entail" in text_lower:
                return "0"
            elif "neutral" in text_lower:
                return "1"
            elif "contradiction" in text_lower or "contradict" in text_lower:
                return "2"

        # For cola task, look for acceptability words
        elif task == "cola":
            if "acceptable" in text_lower or "correct" in text_lower or "grammatical" in text_lower:
                return "1"
            elif "unacceptable" in text_lower or "incorrect" in text_lower or "ungrammatical" in text_lower:
                return "0"

        # For qqp task, look for paraphrase words
        elif task == "qqp":
            if "paraphrase" in text_lower or "same" in text_lower or "similar" in text_lower or "yes" in text_lower:
                return "1"
            elif "not paraphrase" in text_lower or "different" in text_lower or "no" in text_lower:
                return "0"

        # If nothing found, return empty string
        return ""
