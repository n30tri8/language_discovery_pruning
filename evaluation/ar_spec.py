import os
from evaluation.common_evaluation import EvalSpec
from benchmark_loader.datautils import load_paraphrase_arabic
from benchmark_loader.xglue_loader import load_xnli_test
from benchmark_loader.xglue_prompt_templates import SELECTED_ARABIC_TASKS


class AREvalSpec(EvalSpec):
    def __init__(self, benchmark_name):
        super().__init__(SELECTED_ARABIC_TASKS, benchmark_name=benchmark_name, lang='ar')
        self.dataset_base_dir = None

    def set_dataset_base_dir(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir

    def load_eval_data(self, task):
        if task == "xnli":
            # Load XNLI test data for Arabic
            xnli_base_dir = os.path.join(self.dataset_base_dir, "xglue_dataset")
            data = load_xnli_test(xnli_base_dir, lang='ar',
                                sample_size=self.selected_tasks[task]["test_size"])
        elif task == "paraphrase":
            # Load Arabic paraphrasing benchmark data
            data = load_paraphrase_arabic(self.dataset_base_dir,
                                        sample_size=self.selected_tasks[task]["test_size"],
                                        split="test")
        else:
            raise ValueError(f"Task '{task}' is not defined for AREvalSpec")

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

        # ===== XNLI TASK =====
        if task == "xnli":
            # Expected labels: entailment, contradiction, neutral
            valid_labels = ["entailment", "contradiction", "neutral"]

            # Look for bracketed patterns: [entailment], [contradiction], [neutral]
            for label in valid_labels:
                if f"[{label}]" in text:
                    return label

            # Look for quoted patterns: "entailment", "contradiction", "neutral"
            for label in valid_labels:
                if f'"{label}"' in text or f"'{label}'" in text:
                    return label

            # Look for exact label matches (case-insensitive)
            text_lower = text.lower()
            for label in valid_labels:
                if label in text_lower:
                    return label

            # Look for Arabic words indicating entailment, contradiction, or neutral
            # Words indicating entailment (تستلزم/تدل على/تتضمن)
            if any(word in text for word in ["تستلزم", "تدل على", "تتضمن", "تنطوي على", "تشير إلى", "مستلزمة", "متضمنة"]):
                return "entailment"
            # Words indicating contradiction (تتناقض/تخالف/متناقضة)
            elif any(word in text for word in ["تتناقض", "تخالف", "متناقضة", "تعارض", "مخالفة", "متعارضة"]):
                return "contradiction"
            # Words indicating neutral (محايدة/لا علاقة/مستقلة)
            elif any(word in text for word in ["محايدة", "لا علاقة", "مستقلة", "غير مرتبطة", "لا صلة", "محايد"]):
                return "neutral"

        # ===== PARAPHRASE TASK =====
        elif task == "paraphrase":
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

            # Look for Arabic words indicating same or different meaning
            # Words indicating same meaning (نفس المعنى/متشابهتان/مترادفتان) - return 1
            if any(word in text for word in ["نفس المعنى", "متشابهتان", "مترادفتان", "متطابقتان", "نعم", "صحيح", "مماثلتان", "متساويتان"]):
                return "1"
            # Words indicating different meaning (معنى مختلف/مختلفتان/غير متشابهتان) - return 0
            elif any(word in text for word in ["معنى مختلف", "مختلفتان", "غير متشابهتان", "متباينتان", "لا", "خطأ", "مختلفة", "غير متطابقتان"]):
                return "0"

        # If task is not recognized or nothing found, return empty string
        return ""
