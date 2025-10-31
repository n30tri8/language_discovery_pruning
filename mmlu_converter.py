if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import ast
    import re

    # Read parquet file
    df = pd.read_parquet("C:\\Users\\pooya\\Downloads\\test-00000-of-00001 (2).parquet")


    def clean_and_split_choices(raw):
        """Parse 'choices' column into a list of four clean strings (A–D)."""
        # Handle missing values or NaN
        if raw is None:
            return ["", "", "", ""]
        if isinstance(raw, float) and pd.isna(raw):
            return ["", "", "", ""]

        # Handle already-list or numpy array
        if isinstance(raw, (list, np.ndarray)):
            return [str(x).strip() for x in raw] + [""] * (4 - len(raw))

        # Convert everything else to string for parsing
        text = str(raw).strip()

        # Try literal eval if it looks like a list
        if text.startswith("[") and text.endswith("]"):
            try:
                val = ast.literal_eval(text)
                if isinstance(val, (list, np.ndarray)):
                    return [str(x).strip() for x in val] + [""] * (4 - len(val))
            except Exception:
                pass

        # Otherwise manually clean
        text = re.sub(r"[\[\]]", "", text)
        text = text.replace("'", "").replace('"', "")
        # Split by either commas or multiple spaces
        parts = re.split(r"\s{2,}|,\s*|\s(?=[^,]+')", text)
        if len(parts) < 4:
            parts = text.split()

        parts = [p.strip() for p in parts if p.strip()]
        while len(parts) < 4:
            parts.append("")
        return parts[:4]


    def map_answer(num):
        mapping = {0: "A", 1: "B", 2: "C", 3: "D"}
        try:
            return mapping.get(int(num), "")
        except Exception:
            return ""


    # --- Process the dataframe ---

    # Split choices column into A–D safely
    choices_split = df["choices"].apply(clean_and_split_choices)
    df[["A", "B", "C", "D"]] = pd.DataFrame(choices_split.tolist(), index=df.index)

    # Map numeric answers to A–D
    df["Answer"] = df["answer"].apply(map_answer)

    # Reorder and rename columns
    df_final = df.reset_index(drop=True)
    df_final.index += 1  # start IDs from 1
    df_final.insert(0, "id", df_final.index)
    df_final = df_final[["id", "question", "A", "B", "C", "D", "Answer", "subject"]]
    df_final.columns = ["id", "Question", "A", "B", "C", "D", "Answer", "Subject"]

    # Save to CSV
    df_final.to_csv("formatted_mmlu.csv", index=False, encoding="utf-8")

    print("✅ Conversion complete. Saved as formatted_mmlu.csv")
    print(df_final.head(5))
