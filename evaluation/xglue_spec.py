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
            raise ValueError(f"Task '{task}' is not defined for EvalSpec")
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

            # Language-specific patterns
            if self.lang == 'en':
                # English words for entailment
                if any(word in text_lower for word in ["entails", "implies", "follows", "supports", "confirms"]):
                    return "entailment"
                # English words for contradiction
                elif any(word in text_lower for word in ["contradicts", "opposes", "conflicts", "disagrees"]):
                    return "contradiction"
                # English words for neutral
                elif any(word in text_lower for word in ["neutral", "unrelated", "independent", "no relation"]):
                    return "neutral"

            elif self.lang == 'de':
                # German words for entailment (Folgerung/Implikation)
                if any(word in text for word in
                       ["Folgerung", "impliziert", "folgt", "bestätigt", "unterstützt", "schließt"]):
                    return "entailment"
                # German words for contradiction (Widerspruch)
                elif any(word in text for word in ["Widerspruch", "widerspricht", "gegensätzlich", "konfrontiert"]):
                    return "contradiction"
                # German words for neutral (neutral/unabhängig)
                elif any(word in text for word in ["neutral", "unabhängig", "keine Beziehung", "unverbunden"]):
                    return "neutral"

            elif self.lang == 'fr':
                # French words for entailment (implication)
                if any(word in text for word in ["implication", "implique", "entraîne", "confirme", "soutient"]):
                    return "entailment"
                # French words for contradiction (contradiction)
                elif any(word in text for word in ["contradiction", "contredit", "oppose", "conflictuel"]):
                    return "contradiction"
                # French words for neutral (neutre/indépendant)
                elif any(word in text for word in ["neutre", "indépendant", "aucune relation", "non lié"]):
                    return "neutral"

        # ===== PAWSX TASK =====
        elif task == "pawsx":
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

            # Language-specific patterns
            text_lower = text.lower()

            if self.lang == 'en':
                # English words indicating same meaning - return 1
                if any(word in text_lower for word in
                       ["same", "similar", "equivalent", "paraphrase", "yes", "true", "identical"]):
                    return "1"
                # English words indicating different meaning - return 0
                elif any(word in text_lower for word in ["different", "dissimilar", "no", "false", "not", "distinct"]):
                    return "0"

            elif self.lang == 'de':
                # German words indicating same meaning - return 1
                if any(word in text_lower for word in
                       ["gleich", "ähnlich", "gleichwertig", "paraphrase", "ja", "wahr", "identisch", "dasselbe"]):
                    return "1"
                # German words indicating different meaning - return 0
                elif any(word in text_lower for word in
                         ["verschieden", "unterschiedlich", "nein", "falsch", "nicht", "anders"]):
                    return "0"

            elif self.lang == 'fr':
                # French words indicating same meaning - return 1
                if any(word in text_lower for word in
                       ["même", "similaire", "équivalent", "paraphrase", "oui", "vrai", "identique", "pareil"]):
                    return "1"
                # French words indicating different meaning - return 0
                elif any(
                        word in text_lower for word in ["différent", "dissemblable", "non", "faux", "pas", "distinct"]):
                    return "0"

        # If task is not recognized or nothing found, return empty string
        return ""

