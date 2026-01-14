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

system_prompt_fr = """Vous êtes un expert en {field}. Ci-dessous se trouve une question à choix multiples dans ce domaine et ses options de réponse.
Remarque :
(1) Veuillez uniquement fournir l'indice de réponse le plus probable au format : [[Indice de Réponse]], par exemple, si l'option de réponse la plus probable est 'A. Sac à main', alors répondez '[[A]]' ;
(2) Vous devez choisir l'une des options de réponse données 'A, B, C, D' comme étant la plus probable."""

system_prompt_it = """Sei un esperto in {field}. Di seguito è riportata una domanda a scelta multipla in questo campo e le relative opzioni di risposta.
Nota:
(1) Si prega di restituire solo l'indice della risposta più probabile nel formato: [[Indice della risposta]], ad esempio, se l'opzione di risposta più probabile è 'A. Borsa', allora restituire '[[A]]';
(2) È necessario scegliere una delle opzioni di risposta fornite 'A, B, C, D' come risposta più probabile."""

system_prompt_ar = """أنت خبير في {field}. يوجد أدناه سؤال متعدد الخيارات في هذا المجال وخيارات إجابته.
ملحوظة:
(1) يرجى إخراج فهرس الإجابة الأكثر احتمالا فقط بالتنسيق: [[فهرس الإجابة]]، على سبيل المثال، إذا كان خيار الإجابة الأكثر احتمالا هو 'A. حقيبة يد'، فقم بإخراج '[[A]]'؛
(2) يجب عليك اختيار أحد خيارات الإجابة المحددة 'A, B, C, D' باعتبارها الإجابة الأكثر احتمالا."""

system_prompt_hi = """आप {field} के विशेषज्ञ हैं। नीचे इस क्षेत्र में एक बहुविकल्पीय प्रश्न और उसके उत्तर विकल्प दिए गए हैं।
ध्यान दें:
(1) कृपया केवल सबसे संभावित उत्तर सूचकांक को प्रारूप में आउटपुट करें: [[उत्तर सूचकांक]], उदाहरण के लिए, यदि सबसे संभावित उत्तर विकल्प 'ए. हैंडबैग' है, तो '[[A]]' आउटपुट करें;
(2) आपको दिए गए उत्तर विकल्पों 'A, B, C, D' में से किसी एक को सबसे संभावित उत्तर के रूप में चुनना होगा।"""

system_prompt_ja = """あなたは{field}の専門家です。以下に、この分野の多肢選択問題とその解答選択肢を示します。
注：
(1) 最も可能性の高い解答のインデックスのみを次の形式で出力してください：[[解答インデックス]]。例えば、最も可能性の高い解答選択肢が「A. ハンドバッグ」の場合、「[[A]]」と出力します。
(2) 与えられた解答選択肢「A、B、C、D」の中から、最も可能性の高いものを1つ選択する必要があります。"""

system_prompt_field_translation = {
    'de': {
        "field": {
            "philosophy": "Philosophie",
            "professional_law": "Rechtswissenschaften",
            "high_school_mathematics": "Gymnasialmathematik",
            "professional_psychology": "Berufspsychologie",
            "international_law": "Internationales Recht",
            "professional_medicine": "Fachmedizin",
            "sociology": "Soziologie",
            "marketing": "Marketing",
            "high_school_chemistry": "Gymnasialchemie",
            "clinical_knowledge": "Klinisches Wissen",
        }
    },
    'fr': {
        "field": {
            "philosophy": "Philosophie",
            "professional_law": "Droit professionnel",
            "high_school_mathematics": "Mathématiques de lycée",
            "professional_psychology": "Psychologie professionnelle",
            "international_law": "Droit international",
            "professional_medicine": "Médecine professionnelle",
            "sociology": "Sociologie",
            "marketing": "Marketing",
            "high_school_chemistry": "Chimie de lycée",
            "clinical_knowledge": "Connaissances cliniques",
        }
    },
    'it': {
        "field": {
            "philosophy": "Filosofia",
            "professional_law": "Diritto professionale",
            "high_school_mathematics": "Matematica per la scuola superiore",
            "professional_psychology": "Psicologia professionale",
            "international_law": "Diritto internazionale",
            "professional_medicine": "Medicina professionale",
            "sociology": "Sociologia",
            "marketing": "Marketing",
            "high_school_chemistry": "Chimica per la scuola superiore",
            "clinical_knowledge": "Conoscenza clinica",
        }
    },
    'ar': {
        "field": {
            "philosophy": "الفلسفة",
            "professional_law": "القانون المهني",
            "high_school_mathematics": "رياضيات المدرسة الثانوية",
            "professional_psychology": "علم النفس المهني",
            "international_law": "القانون الدولي",
            "professional_medicine": "الطب المهني",
            "sociology": "علم الاجتماع",
            "marketing": "التسويق",
            "high_school_chemistry": "كيمياء المدرسة الثانوية",
            "clinical_knowledge": "المعرفة السريرية",
        }
    },
    'hi': {
        "field": {
            "philosophy": "दर्शन",
            "professional_law": "व्यावसायिक कानून",
            "high_school_mathematics": "उच्च विद्यालय गणित",
            "professional_psychology": "व्यावसायिक मनोविज्ञान",
            "international_law": "अंतर्राष्ट्रीय कानून",
            "professional_medicine": "व्यावसायिक चिकित्सा",
            "sociology": "समाजशास्त्र",
            "marketing": "विपणन",
            "high_school_chemistry": "उच्च विद्यालय रसायन विज्ञान",
            "clinical_knowledge": "नैदानिक ज्ञान",
        }
    },
    'ja': {
        "field": {
            "philosophy": "哲学",
            "professional_law": "専門法学",
            "high_school_mathematics": "高校数学",
            "professional_psychology": "専門心理学",
            "international_law": "国際法",
            "professional_medicine": "専門医学",
            "sociology": "社会学",
            "marketing": "マーケティング",
            "high_school_chemistry": "高校化学",
            "clinical_knowledge": "臨床知識",
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
user_prompt_fr = (
    "[Question]\n{question}\n\n"
    "[Réponses possibles]\n"
    "A. {options[0]}\n"
    "B. {options[1]}\n"
    "C. {options[2]}\n"
    "D. {options[3]}"
)
user_prompt_it = (
    "[Domanda]\n{question}\n\n"
    "[Risposte possibili]\n"
    "A. {options[0]}\n"
    "B. {options[1]}\n"
    "C. {options[2]}\n"
    "D. {options[3]}"
)
user_prompt_ar = (
    "[سؤال]\n{question}\n\n"
    "[إجابات محتملة]\n"
    "A. {options[0]}\n"
    "B. {options[1]}\n"
    "C. {options[2]}\n"
    "D. {options[3]}"
)
user_prompt_hi = (
    "[प्रश्न]\n{question}\n\n"
    "[उम्मीदवार उत्तर]\n"
    "A. {options[0]}\n"
    "B. {options[1]}\n"
    "C. {options[2]}\n"
    "D. {options[3]}"
)
user_prompt_ja = (
    "[質問]\n{question}\n\n"
    "[候補の回答]\n"
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
    "fr": {
        "system_template": make_prompt_template(system_prompt_fr,
                                                translation_mapping=system_prompt_field_translation["fr"]),
        "user_template": make_prompt_template(user_prompt_fr)
    },
    "it": {
        "system_template": make_prompt_template(system_prompt_it,
                                                translation_mapping=system_prompt_field_translation["it"]),
        "user_template": make_prompt_template(user_prompt_it)
    },
    "ar": {
        "system_template": make_prompt_template(system_prompt_ar,
                                                translation_mapping=system_prompt_field_translation["ar"]),
        "user_template": make_prompt_template(user_prompt_ar)
    },
    "hi": {
        "system_template": make_prompt_template(system_prompt_hi,
                                                translation_mapping=system_prompt_field_translation["hi"]),
        "user_template": make_prompt_template(user_prompt_hi)
    },
    "ja": {
        "system_template": make_prompt_template(system_prompt_ja,
                                                translation_mapping=system_prompt_field_translation["ja"]),
        "user_template": make_prompt_template(user_prompt_ja)
    },
}
