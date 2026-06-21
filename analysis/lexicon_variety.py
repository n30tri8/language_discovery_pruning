import csv
import spacy
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# -------------------------
# USER-SPECIFIED VARIABLES
# -------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
LANGUAGES = ["en", "de", "fr", "ar", "hi", "it"]  # Languages to analyze
SUBJECTS = ["philosophy", "international_law", "high_school_mathematics", "professional_psychology",
            "professional_medicine", "sociology", "marketing", "high_school_chemistry", "clinical_knowledge"]  # Subjects to analyze
MAX_QUESTIONS = 300        # Only consider first N questions for each subject

# Language models to use based on file name or explicit override
# Add more language mappings as needed.
LANGUAGE_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "ar": "xx_ent_wiki_sm", # no specific model available "ar_core_news_sm",
    "hi": "xx_ent_wiki_sm", # no specific model available "hi_core_web_sm",
    "it": "it_core_news_sm",
}

# Optionally force a spacy model (set to None to auto-detect from file name)
# FORCE_LANGUAGE = "en"  # e.g. "de" or "en"
FORCE_LANGUAGE: str | None = None  # e.g. "de" or "en"


# -------------------------
# LOAD SPACY MODEL
# -------------------------

def resolve_language_code(filename: str) -> str:
    if FORCE_LANGUAGE is not None:
        return FORCE_LANGUAGE

    # infer by file name prefix (very simple heuristic: "en.csv", "de.csv")
    lang = Path(filename).stem.split("_")[0]
    if lang not in LANGUAGE_MODELS:
        print(f"Could not detect language from file name '{filename}'. Defaulting to English.")
        return "en"

    return lang


def load_spacy_model(filename: str):
    lang = resolve_language_code(filename)
    assert isinstance(lang, str)

    if lang not in LANGUAGE_MODELS:
        print(f"Warning: No spaCy model configured for language '{lang}'. Using a blank pipeline instead.")
        try:
            return spacy.blank(lang)
        except Exception:
            return spacy.blank("xx")

    model_name = LANGUAGE_MODELS[lang]

    print(f"Loading spaCy model: {model_name} ...")
    try:
        return spacy.load(model_name)
    except OSError as exc:
        print(f"Warning: Could not load spaCy model '{model_name}' for language '{lang}': {exc}")
        print("Falling back to a blank pipeline; lemma counts will use token text when needed.")
        try:
            return spacy.blank(lang)
        except Exception:
            return spacy.blank("xx")


# -------------------------
# EXTRACT QUESTIONS
# -------------------------

def load_questions_by_subject(file_path, subjects, max_per_subject):
    subject_questions = defaultdict(list)

    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            subj = row["Subject"]
            if subj in subjects:
                if len(subject_questions[subj]) < max_per_subject:
                    subject_questions[subj].append(row["Question"])

            # If all subjects have enough, stop early
            if all(len(subject_questions[s]) >= max_per_subject for s in subjects):
                break

    return subject_questions


# -------------------------
# COMPUTE LEXICAL VARIETY
# -------------------------

def compute_lexical_variety(nlp, questions):
    lemmas = set()
    for q in questions:
        doc = nlp(q)
        for token in doc:
            # Filter punctuation, spaces, numbers
            if token.is_alpha:
                lemma = token.lemma_.lower().strip()
                if not lemma or lemma == "-pron-":
                    lemma = token.text.lower()
                lemmas.add(lemma)
    return len(lemmas), lemmas


# -------------------------
# MAIN PROCESS
# -------------------------

def main():
    # Dictionary to store results: {subject: {lang: unique_count}}
    results = {subject: {} for subject in SUBJECTS}

    print("\n=== Lexical Variety Results (Per Language) ===\n")

    for lang in LANGUAGES:
        print(f"Processing language: {lang}")
        data_file = SCRIPT_DIR.parent / "benchmark_data" / "mmlu" / f"{lang}.csv"

        # Check if file exists
        if not data_file.exists():
            print(f"  Warning: File not found for {lang}: {data_file}")
            continue

        # Load spaCy model for this language
        nlp = load_spacy_model(str(data_file))

        # Load questions for this language
        subject_questions = load_questions_by_subject(data_file, SUBJECTS, MAX_QUESTIONS)

        # Compute lexical variety for each subject
        for subject, questions in subject_questions.items():
            if questions:  # Only process if questions were found
                unique_count, lemmas = compute_lexical_variety(nlp, questions)
                results[subject][lang] = unique_count
                print(f"  {subject}: {unique_count} unique lemmas ({len(questions)} questions)")

        print()

    # Write results to CSV
    output_dir = SCRIPT_DIR.parent / "results" / "unique lexicon count per subject lanauges"
    output_dir.mkdir(parents=True, exist_ok=True)

    current_date = datetime.now().strftime("%Y-%m-%d")
    output_file = output_dir / f"{current_date}.csv"

    print(f"Writing results to: {output_file}")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Subject"] + LANGUAGES)
        writer.writeheader()

        for subject in SUBJECTS:
            row = {"Subject": subject}
            for lang in LANGUAGES:
                row[lang] = results[subject].get(lang, "N/A")
            writer.writerow(row)

    print("Done.")


if __name__ == "__main__":
    main()