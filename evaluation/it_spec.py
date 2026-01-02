from evaluation.common_evaluation import EvalSpec
from submodules.SparseLLM.datautils import load_uinauil_textualentailmen
from submodules.SparseLLM.xglue_loader import load_pawsx_italian
from submodules.SparseLLM.xglue_prompt_templates import SELECTED_ITALIAN_TASKS


class ITEvalSpec(EvalSpec):
    def __init__(self, benchmark_name):
        super().__init__(SELECTED_ITALIAN_TASKS, benchmark_name=benchmark_name, lang='it')
        self.dataset_base_dir = None

    def set_dataset_base_dir(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir

    def load_eval_data(self, task):
        if task == "uinauil-textualentailment":
            data = load_uinauil_textualentailmen(self.dataset_base_dir,
                                                 sample_size=self.selected_tasks[task]["test_size"], split="test")
        elif task == "pawsx-translated":
            data = load_pawsx_italian(self.dataset_base_dir, sample_size=self.selected_tasks[task]["test_size"],
                                      split="test")
        else:
            raise ValueError(f"Task '{task}' is not defined for EvalSpec")

        return data

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

        # ===== UINAUIL TEXTUAL ENTAILMENT TASK =====
        if task == "uinauil-textualentailment":
            # Expected labels: 0 (not entailed), 1 (entailed)
            valid_labels = ["0", "1"]

            # Look for bracketed patterns: [0], [1]
            for label in valid_labels:
                if f"[{label}]" in text:
                    return label

            # Look for quoted patterns: "0", "1"
            for label in valid_labels:
                if f'"{label}"' in text or f"'{label}'" in text:
                    return label

            # Look for the label at the beginning of the text
            if text and text[0] in valid_labels:
                return text[0]

            # Look for the last occurrence of any valid label in the text
            for i in range(len(text) - 1, -1, -1):
                if text[i] in valid_labels:
                    return text[i]

            # Look for Italian words indicating entailment or no entailment
            text_lower = text.lower()
            # Words indicating entailment (1)
            if any(word in text_lower for word in
                   ["sì", "si", "vero", "corretto", "implica", "consegue", "deriva", "segue"]):
                return "1"
            # Words indicating no entailment (0)
            elif any(word in text_lower for word in
                     ["no", "falso", "scorretto", "non implica", "non consegue", "non deriva", "non segue"]):
                return "0"

        # ===== PAWSX TRANSLATED TASK =====
        elif task == "pawsx-translated":
            # Expected labels: 0 (different meaning), 1 (same meaning)
            valid_labels = ["0", "1"]

            # Look for bracketed patterns: [0], [1]
            for label in valid_labels:
                if f"[{label}]" in text:
                    return label

            # Look for quoted patterns: "0", "1"
            for label in valid_labels:
                if f'"{label}"' in text or f"'{label}'" in text:
                    return label

            # Look for the label at the beginning of the text
            if text and text[0] in valid_labels:
                return text[0]

            # Look for the last occurrence of any valid label in the text
            for i in range(len(text) - 1, -1, -1):
                if text[i] in valid_labels:
                    return text[i]

            # Look for Italian words indicating same or different meaning
            text_lower = text.lower()
            # Words indicating same meaning (1)
            if any(word in text_lower for word in
                   ["stesso", "uguale", "identico", "stesso significato", "medesimo", "equivalente", "sì", "si"]):
                return "1"
            # Words indicating different meaning (0)
            elif any(word in text_lower for word in
                     ["diverso", "differente", "diverso significato", "distinto", "differisce", "no"]):
                return "0"

        # If task is not recognized or nothing found, return empty string
        return ""
