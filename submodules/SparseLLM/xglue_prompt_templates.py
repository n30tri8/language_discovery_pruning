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

xnli_system_prompt_fr = """
Vous êtes un système d'inférence en langage naturel.
Étant donné une prémisse et une hypothèse, déterminez si l'hypothèse est impliquée,
contradictoire ou neutre. Répondez par l'un des suivants : entailment, contradiction, neutral.
"""

xnli_user_prompt_fr = """
Prémisse : {premise}
Hypothèse : {hypothesis}
Quelle est la relation ?
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
{label}
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

pawsx_system_prompt_fr = """
Vous êtes un système d'identification de paraphrases.
Retournez 1 si les deux phrases ont le même sens, sinon 0.
"""

pawsx_user_prompt_fr = """
Phrase 1 : {sentence1}
Phrase 2 : {sentence2}
Ces phrases ont-elles le même sens ?
"""

# ===== Italian =====
xnli_system_prompt_it = """
Sei un sistema di inferenza del linguaggio naturale.
Data una premessa e un'ipotesi, decidi se l'ipotesi è implicata, contraddetta o neutra.
Rispondi con uno di: entailment, contradiction, neutral.
"""

xnli_user_prompt_it = """
Premessa: {premise}
Ipotesi: {hypothesis}
Qual è la relazione?
"""

pawsx_system_prompt_it = """
Sei un sistema di identificazione di parafrasi.
Restituisci 1 se le due frasi hanno lo stesso significato, altrimenti 0.
"""

pawsx_user_prompt_it = """
Frase 1: {sentence1}
Frase 2: {sentence2}
Queste frasi hanno lo stesso significato?
"""

# ===== Hindi =====
xnli_system_prompt_hi = """
आप एक प्राकृतिक भाषा अनुमान प्रणाली हैं。
एक आधार और एक परिकल्पना को देखते हुए, यह तय करें कि परिकल्पना निहित है, विरोधाभासी है, या तटस्थ है。
इनमें से किसी एक के साथ उत्तर दें: entailment, contradiction, neutral。
"""

xnli_user_prompt_hi = """
आधार: {premise}
परिकल्पना: {hypothesis}
क्या संबंध है?
"""

pawsx_system_prompt_hi = """
आप एक पैराफ्रेज पहचान प्रणाली हैं。
यदि दोनों वाक्यों का एक ही अर्थ है तो 1 लौटाएं, अन्यथा 0。
"""

pawsx_user_prompt_hi = """
वाक्य 1: {sentence1}
वाक्य 2: {sentence2}
क्या इन वाक्यों का एक ही अर्थ है?
"""

# ===== Arabic =====
xnli_system_prompt_ar = """
أنت نظام استدلال لغوي طبيعي.
بالنظر إلى فرضية ونظرية، قرر ما إذا كانت النظرية مستلزمة أو متناقضة أو محايدة.
أجب بواحد مما يلي: entailment, contradiction, neutral.
"""

xnli_user_prompt_ar = """
الفرضية: {premise}
النظرية: {hypothesis}
ما هي العلاقة؟
"""

pawsx_system_prompt_ar = """
أنت نظام تعريف إعادة الصياغة.
أرجع 1 إذا كانت الجملتان تعنيان نفس الشيء، وإلا فأرجع 0.
"""

pawsx_user_prompt_ar = """
الجملة 1: {sentence1}
الجملة 2: {sentence2}
هل هاتان الجملتان لهما نفس المعنى؟
"""

# ===== Japanese =====
xnli_system_prompt_ja = """
あなたは自然言語推論システムです。
前提と仮説が与えられた場合、仮説が含意、矛盾、または中立のいずれであるかを判断します。
次の中から1つ選択して回答してください: entailment, contradiction, neutral。
"""

xnli_user_prompt_ja = """
前提: {premise}
仮説: {hypothesis}
関係は何ですか？
"""

pawsx_system_prompt_ja = """
あなたは言い換え識別システムです。
2つの文が同じ意味を持つ場合は1を、そうでない場合は0を返します。
"""

pawsx_user_prompt_ja = """
文1: {sentence1}
文2: {sentence2}
これらの文は同じ意味ですか？
"""

SELECTED_XGLUE_TASKS = {
    "xnli": {
        "sample_size": 500,
        "test_size": 50,
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
        "fr": {
            "system_template": make_prompt_template(xnli_system_prompt_fr),
            "user_template": make_prompt_template(xnli_user_prompt_fr),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
        "it": {
            "system_template": make_prompt_template(xnli_system_prompt_it),
            "user_template": make_prompt_template(xnli_user_prompt_it),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
        "hi": {
            "system_template": make_prompt_template(xnli_system_prompt_hi),
            "user_template": make_prompt_template(xnli_user_prompt_hi),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
        "ar": {
            "system_template": make_prompt_template(xnli_system_prompt_ar),
            "user_template": make_prompt_template(xnli_user_prompt_ar),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
        "ja": {
            "system_template": make_prompt_template(xnli_system_prompt_ja),
            "user_template": make_prompt_template(xnli_user_prompt_ja),
            "assistant_template": make_prompt_template(xnli_assistant_prompt_all)
        },
    },
    "pawsx": {
        "sample_size": 500,
        "test_size": 50,
        "en": {
            "system_template": make_prompt_template(pawsx_system_prompt_en),
            "user_template": make_prompt_template(pawsx_user_prompt_en),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "de": {
            "system_template": make_prompt_template(pawsx_system_prompt_de),
            "user_template": make_prompt_template(pawsx_user_prompt_de),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "fr": {
            "system_template": make_prompt_template(pawsx_system_prompt_fr),
            "user_template": make_prompt_template(pawsx_user_prompt_fr),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "it": {
            "system_template": make_prompt_template(pawsx_system_prompt_it),
            "user_template": make_prompt_template(pawsx_user_prompt_it),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "hi": {
            "system_template": make_prompt_template(pawsx_system_prompt_hi),
            "user_template": make_prompt_template(pawsx_user_prompt_hi),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "ar": {
            "system_template": make_prompt_template(pawsx_system_prompt_ar),
            "user_template": make_prompt_template(pawsx_user_prompt_ar),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        },
        "ja": {
            "system_template": make_prompt_template(pawsx_system_prompt_ja),
            "user_template": make_prompt_template(pawsx_user_prompt_ja),
            "assistant_template": make_prompt_template(pawsx_assistant_prompt_all),
        }
    }
}
