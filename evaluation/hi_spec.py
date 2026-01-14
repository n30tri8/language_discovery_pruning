import os

from evaluation.common_evaluation import EvalSpec
from benchmark_loader.datautils import load_paraphrase_hindi
from benchmark_loader.xglue_loader import load_xnli_test
from benchmark_loader.xglue_prompt_templates import SELECTED_HINDI_TASKS


class HIEvalSpec(EvalSpec):
    def __init__(self, benchmark_name):
        super().__init__(SELECTED_HINDI_TASKS, benchmark_name=benchmark_name, lang='hi')
        self.dataset_base_dir = None

    def set_dataset_base_dir(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir

    def load_eval_data(self, task):
        if task == "xnli":
            # Load XNLI test data for Hindi
            xnli_base_dir = os.path.join(self.dataset_base_dir, "xglue_dataset")
            data = load_xnli_test(xnli_base_dir, lang='hi',
                                  sample_size=self.selected_tasks[task]["test_size"])
        elif task == "paraphrase":
            # Load Hindi paraphrasing benchmark data
            data = load_paraphrase_hindi(self.dataset_base_dir,
                                          sample_size=self.selected_tasks[task]["test_size"],
                                          split="test")
        else:
            raise ValueError(f"Task '{task}' is not defined for HIEvalSpec")

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

            # Look for Hindi words indicating entailment, contradiction, or neutral
            # Words indicating entailment (निहित/सत्य/समर्थन/अनुसरण)
            if any(word in text for word in ["निहित", "सत्य", "समर्थन", "अनुसरण", "सिद्ध", "प्रमाणित", "दर्शाता", "निकलता", "अभिप्राय"]):
                return "entailment"
            # Words indicating contradiction (विरोधाभास/झूठ/असत्य/विपरीत)
            elif any(word in text for word in ["विरोधाभास", "झूठ", "असत्य", "विपरीत", "गलत", "खंडन", "विरोधी", "मतभेद", "संघर्ष"]):
                return "contradiction"
            # Words indicating neutral (तटस्थ/अज्ञात/स्पष्ट नहीं/कोई संबंध नहीं)
            elif any(word in text for word in ["तटस्थ", "अज्ञात", "स्पष्ट नहीं", "कोई संबंध नहीं", "निष्पक्ष", "अलग", "असंबंधित", "स्वतंत्र"]):
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

            # Look for Hindi words indicating same or different meaning
            # Words indicating same meaning (समान अर्थ/एक ही अर्थ/सही/हाँ) - return 1
            if any(word in text for word in ["समान अर्थ", "एक ही अर्थ", "सही", "हाँ", "हां", "सत्य", "बराबर", "समानता", "सामान", "मिलता"]):
                return "1"
            # Words indicating different meaning (अलग अर्थ/भिन्न अर्थ/गलत/नहीं) - return 0
            elif any(word in text for word in ["अलग अर्थ", "भिन्न अर्थ", "गलत", "नहीं", "असत्य", "विभिन्न", "अलग", "फर्क", "मतभेद"]):
                return "0"

        # If task is not recognized or nothing found, return empty string
        return ""
