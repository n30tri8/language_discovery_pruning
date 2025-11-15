from typing import Callable, Mapping


# todo, duplicat in prompt_templates.py
class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        # leave unknown placeholders unchanged
        return "{" + key + "}"


def make_prompt_template(prompt: str) -> Callable[[Mapping], str]:
    """
    Return a builder function that formats `prompt` by replacing {key}
    placeholders using values from a provided mapping. Missing keys remain
    as their `{key}` placeholder.
    """

    def builder(values: Mapping) -> str:
        # ensure values are strings to avoid formatting errors
        string_values = {k: str(v) for k, v in values.items()}
        return prompt.format_map(_SafeDict(string_values))

    return builder


############################
#  XGLUE PROMPT TEMPLATES  #
############################

# ===== XNLI =====
xnli_system_prompt_en = """
You are a natural language inference system.
Given a premise and a hypothesis, decide whether the hypothesis is entailed,
contradicted, or neutral. Answer with one of: entailment, contradiction, neutral.
"""

xnli_user_prompt_en = """
Premise: {premise}
Hypothesis: {hypothesis}
What is the relationship?
"""

xnli_assistant_prompt_all = """
{label}
"""

xnli_system_prompt_de = """
Sie sind ein System zur natürlichen Sprachinferenz.
Gegeben eine Prämisse und eine Hypothese, entscheiden Sie, ob die Hypothese impliziert,
widersprochen oder neutral ist. Antworten Sie mit einem der folgenden: entailment, contradiction, neutral.
"""

xnli_user_prompt_de = """
Prämisse: {premise}
Hypothese: {hypothesis}
Was ist die Beziehung?
"""

# ===== PAWS-X =====
pawsx_system_prompt_en = """
You are a paraphrase identification system.
Return 1 if the two sentences mean the same thing, otherwise 0.
"""

pawsx_user_prompt_en = """
Sentence 1: {sentence1}
Sentence 2: {sentence2}
Do these sentences have the same meaning?
"""

pawsx_assistant_prompt_all = """
{label_text}
"""

pawsx_system_prompt_de = """
Sie sind ein System zur Identifikation von Paraphrasen.
Antworten Sie mit 1, wenn die beiden Sätze dasselbe bedeuten, andernfalls mit 0.
"""

pawsx_user_prompt_de = """
Satz 1: {sentence1}
Satz 2: {sentence2}
Haben diese Sätze dieselbe Bedeutung?
"""

# Update SELECTED_XGLUE_TASKS to include German (DE) for XNLI and PAWS-X
SELECTED_XGLUE_TASKS = {
    "xnli": {
        "sample_size": 500,
        "en": {
            "system_template": make_prompt_template(xnli_system_prompt_en),
            "user_template": make_prompt_template(xnli_user_prompt_en),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
        "de": {
            "system_template": make_prompt_template(xnli_system_prompt_de),
            "user_template": make_prompt_template(xnli_user_prompt_de),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
    },
    "pawsx": {
        "sample_size": 500,
        "en": {
            "system_template": make_prompt_template(pawsx_system_prompt_en),
            "user_template": make_prompt_template(pawsx_user_prompt_en),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "de": {
            "system_template": make_prompt_template(pawsx_system_prompt_de),
            "user_template": make_prompt_template(pawsx_user_prompt_de),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        }
    }
}
