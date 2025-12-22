import csv
import spacy
from collections import defaultdict
from pathlib import Path

# -------------------------
# USER-SPECIFIED VARIABLES
# -------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR.parent / "benchmark_data" / "mmlu" / "en.csv"
SUBJECTS = ["philosophy", "professional_law", "high_school_mathematics", "professional_psychology"]  # Subjects to analyze
MAX_QUESTIONS = 500        # Only consider first N questions for each subject

# Language models to use based on file name or explicit override
# Add more language mappings as needed.
LANGUAGE_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
}

# Optionally force a spacy model (set to None to auto-detect from file name)
FORCE_LANGUAGE = "en"  # e.g. "de" or "en"


# -------------------------
# LOAD SPACY MODEL
# -------------------------

def load_spacy_model(filename):
    # choose lang from explicit override
    if FORCE_LANGUAGE:
        lang = FORCE_LANGUAGE
    else:
        # infer by file name prefix (very simple heuristic: "en.csv", "de.csv")
        lang = Path(filename).stem.split("_")[0]
        if lang not in LANGUAGE_MODELS:
            print(f"Could not detect language from file name '{filename}'. Defaulting to English.")
            lang = "en"

    model_name = LANGUAGE_MODELS[lang]
    print(f"Loading spaCy model: {model_name} ...")
    return spacy.load(model_name)


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
                lemmas.add(token.lemma_.lower())
    return len(lemmas), lemmas


# -------------------------
# MAIN PROCESS
# -------------------------

def main():
    nlp = load_spacy_model(DATA_FILE)
    subject_questions = load_questions_by_subject(DATA_FILE, SUBJECTS, MAX_QUESTIONS)

    print("\n=== Lexical Variety Results ===\n")

    for subject, questions in subject_questions.items():
        unique_count, lemmas = compute_lexical_variety(nlp, questions)
        print(f"Subject: {subject}")
        print(f"  Questions analyzed: {len(questions)}")
        print(f"  Unique lemmas: {unique_count}\n")

        # If you want to inspect the lemmas, uncomment:
        # print(lemmas)
        # print()

    print("Done.")


if __name__ == "__main__":
    main()