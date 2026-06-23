philosophical_essay_generation_template = {
    'en':
"""This is a test of philosophical understanding and reasoning ability.
Write a concise essay (300–400 words) that demonstrates your understanding of the following philosophical concepts:
1. Empiricism
2. Rationalism
3. Utilitarianism
4. Existentialism
Do not define these concepts independently. Instead:
- Explain each concept briefly.
- Discuss how the concepts relate to one another.
- Identify at least one philosophical tension or disagreement involving these concepts.
- Present a coherent perspective that integrates all concepts.
- Use precise and academically appropriate language.
The goal is not to summarize philosophical history but to demonstrate conceptual understanding, reasoning, and synthesis.""",


    'de':
"""Dies ist ein Test des philosophischen Verständnisses und der Argumentationsfähigkeit.
Schreiben Sie einen prägnanten Essay (300–400 Wörter), der Ihr Verständnis für die folgenden philosophischen Konzepte zeigt:
1. Empirismus
2. Rationalismus
3. Utilitarismus
4. Existenzialismus
Definieren Sie diese Konzepte nicht unabhängig voneinander. Stattdessen:
- Erklären Sie jedes Konzept kurz.
- Diskutieren Sie, wie die Konzepte miteinander zusammenhängen.
- Identifizieren Sie mindestens eine philosophische Spannung oder Uneinigkeit, die diese Konzepte betrifft.
- Präsentieren Sie eine kohärente Perspektive, die alle Konzepte integriert.
- Verwenden Sie präzise und akademisch angemessene Sprache.
Das Ziel ist nicht, die philosophische Geschichte zusammenzufassen, sondern konzeptionelles Verständnis, Argumentation und Synthese zu demonstrieren.""",


    'fr':
"""Il s'agit d'un test de compréhension et de capacité de raisonnement philosophique.
Rédigez un essai concis (300 à 400 mots) qui démontre votre compréhension des concepts philosophiques suivants :
1. Empirisme
2. Rationalisme
3. Utilitarisme
4. Existentialisme
Ne définissez pas ces concepts indépendamment. Au lieu de cela :
- Expliquez brièvement chaque concept.
- Discutez de la façon dont les concepts se rapportent les uns aux autres.
- Identifiez au moins une tension ou un désaccord philosophique impliquant ces concepts.
- Présentez une perspective cohérente qui intègre tous les concepts.
- Utilisez un langage précis et académiquement approprié.
L'objectif n'est pas de résumer l'histoire de la philosophie mais de démontrer une compréhension, un raisonnement et une synthèse conceptuelle.""",


    'it':
"""Questo è un test di comprensione e capacità di ragionamento filosofico.
Scrivi un saggio conciso (300-400 parole) che dimostri la tua comprensione dei seguenti concetti filosofici:
1. Empirismo
2. Razionalismo
3. Utilitarismo
4. Esistenzialismo
Non definire questi concetti in modo indipendente. Invece:
- Spiega brevemente ogni concetto.
- Discuti come i concetti si relazionano tra loro.
- Identifica almeno una tensione o disaccordo filosofico che coinvolge questi concetti.
- Presenta una prospettiva coerente che integri tutti i concetti.
- Usa un linguaggio preciso e accademicamente appropriato.
L'obiettivo non è riassumere la storia filosofica ma dimostrare comprensione concettuale, ragionamento e sintesi.""",


    'ar':
"""هذا اختبار للفهم والقدرة على الاستدلال الفلسفي.
اكتب مقالاً موجزاً (300-400 كلمة) يوضح فهمك للمفاهيم الفلسفية التالية:
1. التجريبية
2. العقلانية
3. النفعية
4. الوجودية
لا تقم بتعريف هذه المفاهيم بشكل مستقل. بدلاً من ذلك:
- اشرح كل مفهوم باختصار.
- ناقش كيف ترتبط المفاهيم ببعضها البعض.
- حدد توتراً أو خلافاً فلسفياً واحداً على الأقل يتضمن هذه المفاهيم.
- قدم منظوراً متماسكاً يدمج كل المفاهيم.
- استخدم لغة دقيقة ومناسبة أكاديمياً.
الهدف ليس تلخيص التاريخ الفلسفي بل إثبات الفهم المفاهيمي والاستدلال والقدرة على التركيب.""",


    'hi':
"""यह दार्शनिक समझ और तर्क क्षमता का परीक्षण है।
एक संक्षिप्त निबंध (300-400 शब्द) लिखें जो निम्नलिखित दार्शनिक अवधारणाओं की आपकी समझ को प्रदर्शित करे:
1. अनुभववाद
2. बुद्धिवाद
3. उपयोगितावाद
4. अस्तित्ववाद
इन अवधारणाओं को स्वतंत्र रूप से परिभाषित न करें। इसके बजाय:
- प्रत्येक अवधारणा को संक्षेप में समझाएं।
- चर्चा करें कि अवधारणाएं एक दूसरे से कैसे संबंधित हैं।
- इन अवधारणाओं से जुड़े कम से कम एक दार्शनिक तनाव या असहमति की पहचान करें।
- एक सुसंगत दृष्टिकोण प्रस्तुत करें जो सभी अवधारणाओं को एकीकृत करता हो।
- सटीक और अकादमिक रूप से उपयुक्त भाषा का उपयोग करें।
लक्ष्य दार्शनिक इतिहास को संक्षेप में प्रस्तुत करना नहीं है, बल्कि वैचारिक समझ, तर्क और संश्लेषण का प्रदर्शन करना है।"""
}

# min_new_tokens=
# new_tokens_generation_bounds = {
#     "en": {"max_new_tokens": 512, "min_new_tokens": 50},
#     "it": {"max_new_tokens": 750, "min_new_tokens": 75},   # Western European multiplier
#     "fr": {"max_new_tokens": 750, "min_new_tokens": 75},
#     "de": {"max_new_tokens": 750, "min_new_tokens": 75},
#     "ar": {"max_new_tokens": 1800, "min_new_tokens": 150}, # Arabic/Persian script adjustment
#     "hi": {"max_new_tokens": 1800, "min_new_tokens": 150},
# }