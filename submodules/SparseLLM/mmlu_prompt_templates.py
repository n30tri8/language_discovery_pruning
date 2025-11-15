from typing import Callable, Mapping, Optional


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        # leave unknown placeholders unchanged
        return "{" + key + "}"


def make_prompt_template(prompt: str, translation_mapping: Optional[Mapping] = None) -> Callable[[Mapping], str]:
    """
    Return a builder function that formats `prompt` by replacing {key}
    placeholders using values from a provided mapping. Missing keys remain
    as their `{key}` placeholder.
    example for translation_mapping
    translation_mapping = {
        "field": {
            "philosophy": "Philosophie",
            "professional_law": "Rechtswissenschaften",
            "high_school_mathematics": "Gymnasialmathematik",
            "professional_psychology": "Berufspsychologie",
        }
    }
    """

    def builder(values: Mapping) -> str:
        if translation_mapping is not None:
            for fields_to_be_translated in translation_mapping.keys():
                if fields_to_be_translated in values:
                    raw_value = values[fields_to_be_translated]
                    values[fields_to_be_translated] = translation_mapping[fields_to_be_translated][raw_value]

        return prompt.format_map(_SafeDict(values))

    return builder


## system prompts
system_prompt_en = """You are an expert in {field}. Below is a multiple-choice question in this field and its answer options.
Note:
(1) Please only output the most likely answer index in the format: [[Answer Index]], for example, if the most likely answer option is 'A. Handbag', then output '[[A]]';
(2) You must choose one of the given answer options 'A, B, C, D' as the most likely answer."""

system_prompt_de = """Sie sind ein Experte in {field}. Unten finden Sie eine Multiple-Choice-Frage in diesem Bereich und die dazugehörigen Antwortmöglichkeiten.\nHinweis:\n(1) Bitte geben Sie nur den wahrscheinlichsten Antwortindex im Format: [[Antwortindex]] aus. Wenn die wahrscheinlichste Antwortoption beispielsweise 'A. Handtasche' ist, geben Sie '[[A]]' aus;\n(2) Sie müssen eine der angegebenen Antwortoptionen 'A, B, C, D' als die wahrscheinlichste auswählen."""

system_prompt_field_translation = {
    'de': {
        "field": {
            "philosophy": "Philosophie",
            "professional_law": "Rechtswissenschaften",
            "high_school_mathematics": "Gymnasialmathematik",
            "professional_psychology": "Berufspsychologie",
        }
    }
}
## user prompts
user_prompt_en = (
    "[Question]\n{question}\n\n"
    "[Candidate Answers]\n"
    "A. {options[0]}\n"
    "B. {options[1]}\n"
    "C. {options[2]}\n"
    "D. {options[3]}"
)
user_prompt_de = (
    "[Frage]\n{question}\n\n"
    "[Antwortmöglichkeiten]\n"
    "A. {options[0]}\n"
    "B. {options[1]}\n"
    "C. {options[2]}\n"
    "D. {options[3]}"
)

MMMLU_PROMPT = {
    "en": {
        "system_template": make_prompt_template(system_prompt_en),
        "user_template": make_prompt_template(user_prompt_en)
    },
    "de": {
        "system_template": make_prompt_template(system_prompt_de,
                                                translation_mapping=system_prompt_field_translation["de"]),
        "user_template": make_prompt_template(user_prompt_de)
    },
}
