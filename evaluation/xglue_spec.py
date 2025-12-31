from evaluation.common_evaluation import EvalSpec
from submodules.SparseLLM.xglue_loader import load_xnli_test, load_pawsx_test
from submodules.SparseLLM.xglue_prompt_templates import SELECTED_XGLUE_TASKS


class XGlueEvalSpec(EvalSpec):
    def __init__(self, benchmark_name, lang):
        super().__init__(SELECTED_XGLUE_TASKS, benchmark_name=benchmark_name, lang=lang)
        self.dataset_base_dir = None

    def set_dataset_base_dir(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir

    def load_eval_data(self, task):
        if task == "xnli":
            loader = load_xnli_test
        elif task == "pawsx":
            loader = load_pawsx_test
        else:
            raise ValueError(f"Task '{task}' is not defined for XGlueEvalSpec")
        data = loader(self.dataset_base_dir, self.lang, sample_size=self.selected_tasks[task]["test_size"],
                      split="test")

        return data

    def build_system_prompt(self, task, record):
        sys_prompt = self.selected_tasks[task][self.lang]["system_template"](record)

        return sys_prompt

    def build_user_prompt(self, task, record):
        user_prompt = self.selected_tasks[task][self.lang]["user_template"](record)

        return user_prompt

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

        # Define valid labels for each XGLUE task
        valid_labels = {
            "xnli": ["entailment", "contradiction", "neutral"],  # Text labels for XNLI
            "pawsx": ["0", "1"]  # Numeric labels for PAWS-X (0=different meaning, 1=same meaning)
        }

        task_labels = valid_labels[task]

        # For XNLI task - look for text-based labels
        if task == "xnli":
            # Look for bracketed patterns first: [entailment], [neutral], [contradiction]
            for label in task_labels:
                if f"[{label}]" in text.lower():
                    return label

            # Look for quoted patterns: "entailment", 'neutral', etc.
            for label in task_labels:
                if f'"{label}"' in text.lower() or f"'{label}'" in text.lower():
                    return label

            # Look for the labels directly in the text (case insensitive)
            text_lower = text.lower()
            for label in task_labels:
                if label in text_lower:
                    return label

        # For PAWS-X task - look for numeric labels
        elif task == "pawsx":
            # Look for bracketed patterns: [0], [1]
            for label in task_labels:
                if f"[{label}]" in text:
                    return label

            # Look for quoted patterns: "0", "1"
            for label in task_labels:
                if f'"{label}"' in text or f"'{label}'" in text:
                    return label

            # Look for the label at the beginning of the text
            if text and text[0] in task_labels:
                return text[0]

            # Look for the last occurrence of any valid label in the text
            for i in range(len(text) - 1, -1, -1):
                if text[i] in task_labels:
                    return text[i]

            # Look for words that might indicate the answer
            text_lower = text.lower()
            if "same" in text_lower or "yes" in text_lower or "paraphrase" in text_lower or "similar" in text_lower:
                return "1"
            elif "different" in text_lower or "no" in text_lower or "not" in text_lower:
                return "0"

        # If nothing found, return empty string
        return ""
