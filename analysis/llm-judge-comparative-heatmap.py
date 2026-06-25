import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


languages = ["EN", "DE", "FR", "IT", "AR", "HI"]

# Initialize matrix
n = len(languages)
matrix = np.eye(n)  # diagonal = 1
# Fill pairwise scores
scores = {
    ("EN", "DE"): 1,
    ("EN", "FR"): 0.88,
    ("EN", "IT"): -0.35,
    ("EN", "AR"): 0.98,
    ("EN", "HI"): 1,

    ("DE", "FR"): 0.25,
    ("DE", "IT"): -0.82,
    ("DE", "AR"): 0.57,
    ("DE", "HI"): 0.78,

    ("FR", "IT"): -0.95,
    ("FR", "AR"): 0.48,
    ("FR", "HI"): 0.26,

    ("IT", "AR"): 0.98,
    ("IT", "HI"): 1.0,

    ("AR", "HI"): -0.30
}

# Fill symmetric matrix
lang_to_idx = {lang: i for i, lang in enumerate(languages)}

for (a, b), value in scores.items():
    i = lang_to_idx[a]
    j = lang_to_idx[b]

    matrix[i, j] = value
    matrix[j, i] = value

df = pd.DataFrame(matrix, index=languages, columns=languages)

# mask the upper triangle (including diagonal) so only the lower half is shown
mask = np.triu(np.ones_like(df, dtype=bool))

plt.figure(figsize=(7, 6))

# pass the mask to seaborn so the upper triangle and diagonal are not filled
sns.heatmap(
    df,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "LLM-as-a-judge Comparison Score"}
)

# plt.title("Pairwise Comparison Across Language-Specific-Pruned Networks")
plt.xlabel("First Language in Pair")
plt.ylabel("Second Language in Pair")

plt.tight_layout()
# plt.savefig(
#     "language_similarity_heatmap.pdf",
#     bbox_inches="tight"
# )
plt.show()