# python
from typing import Callable, Mapping

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


# Templates for MNLI (Multi-Genre Natural Language Inference) task
# Updated to include numeric labels and their mapping.

mnli_system_prompt = """
You are an assistant tasked with determining the relationship between a given premise and a hypothesis.
Decide whether the hypothesis is entailed by the premise, contradicts it, or is neutral with respect to it.
Use the following numeric labels: 0 = entailment, 1 = neutral, 2 = contradiction.
Your answer should contain only the number corresponding to the correct label.
"""

mnli_user_prompt = """
Premise: {premise}
Hypothesis: {hypothesis}
What is the numeric relationship between the premise and the hypothesis?
"""

mnli_assistant_prompt = """
{label}
"""


# Templates for CoLA (Corpus of Linguistic Acceptability) task
cola_system_prompt = """
You are an assistant tasked with determining whether a given sentence is linguistically acceptable or not. 
The labels are numeric, and the mapping is as follows: 0 = unacceptable, 1 = acceptable. 
Your answer should only be the number corresponding to the label.
"""

cola_user_prompt = """
Sentence: {sentence}
Is the sentence linguistically acceptable? Provide the numeric label.
"""

cola_assistant_prompt = """
{label}
"""


# Templates for QQP (Quora Question Pairs) task
qqp_system_prompt = """
You are an assistant tasked with determining whether two given questions are paraphrases of each other. 
The labels are numeric, and the mapping is as follows: 0 = not paraphrases, 1 = paraphrases. 
Your answer should only be the number corresponding to the label.
"""

qqp_user_prompt = """
Question 1: {question1}
Question 2: {question2}
Are these questions paraphrases of each other? Provide the numeric label.
"""

qqp_assistant_prompt = """
{label}
"""


# Templates for STS-B (Semantic Textual Similarity Benchmark) task
stsb_system_prompt = """
You are an assistant tasked with determining the semantic similarity between two given sentences. 
Your job is to provide a similarity score on a scale from 0 to 5, where 0 means no similarity and 5 means complete similarity. 
Your answer should only be the numeric score.
"""

stsb_user_prompt = """
Sentence 1: {sentence1}
Sentence 2: {sentence2}
What is the semantic similarity score between these sentences? Provide a score from 0 to 5.
"""

stsb_assistant_prompt = """
{score}
"""


# Define a static set of GLUE tasks for pruning
SELECTED_GLUE_TASKS = {
    "mnli": {
        "system_template": make_prompt_template(mnli_system_prompt),
        "user_template": make_prompt_template(mnli_user_prompt),
        "assistant_template": make_prompt_template(mnli_assistant_prompt),
        # "sample_size": 3000
        "sample_size": 300
    },
    "cola": {
        "system_template": make_prompt_template(cola_system_prompt),
        "user_template": make_prompt_template(cola_user_prompt),
        "assistant_template": make_prompt_template(cola_assistant_prompt),
        # "sample_size": 1000
        "sample_size": 100
    },
    "qqp": {
        "system_template": make_prompt_template(qqp_system_prompt),
        "user_template": make_prompt_template(qqp_user_prompt),
        "assistant_template": make_prompt_template(qqp_assistant_prompt),
        # "sample_size": 2000
        "sample_size": 200
    },
    "stsb": {
        "system_template": make_prompt_template(stsb_system_prompt),
        "user_template": make_prompt_template(stsb_user_prompt),
        "assistant_template": make_prompt_template(stsb_assistant_prompt),
        # "sample_size": 1000
        "sample_size": 100
    }
}