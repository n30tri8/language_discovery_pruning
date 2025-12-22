import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------
# USER INPUT
# --------------------------------------------------------

# Accuracy change data per subject (each list = [EN, DE, FR])
accuracy_changes = {
    "philosophy": [32.51, 42.42, 59.03],
    "professional_law": [38.35, 25.90, 93.49],
    "high_school_mathematics": [32.68, 10.58, 23.65],
    "professional_psychology": [39.22, 29.00, 53.87],
}

# Lemma variety per subject
lemma_variety = {
    "philosophy": 896,
    "professional_law": 4592,
    "high_school_mathematics": 927,
    "professional_psychology": 2123,
}

# --------------------------------------------------------
# PREPARE DATA
# --------------------------------------------------------

subjects = list(accuracy_changes.keys())

# Compute averaged accuracy per subject
avg_accuracy = [np.mean(accuracy_changes[subj]) for subj in subjects]

# Match lemma variety in same order
lemma_counts = [lemma_variety[subj] for subj in subjects]

# Convert to numpy arrays
X = np.array(lemma_counts, dtype=float)
Y = np.array(avg_accuracy, dtype=float)

# ============================
# PEARSON CORRELATION
# ============================
r, p = pearsonr(X, Y)

print("=== Correlation Analysis ===")
print("Subjects:", subjects)
print("Lemma variety (X):", X)
print("Avg accuracy change (Y):", Y)
print(f"\nPearson correlation coefficient (r): {r:.4f}")
print(f"P-value: {p:.6f}")

# ============================
# SPEARMAN CORRELATION
# ============================
spearman_rho, spearman_p = spearmanr(X, Y)
print("\nSpearman correlation (rho):", spearman_rho)
print("Spearman p-value:", spearman_p)

# --------------------------------------------------------
# REGRESSION LINE
# --------------------------------------------------------

slope, intercept, r_val, p_val, stderr = linregress(X, Y)

print("\nLinear regression:")
print(f"  slope:     {slope:.4f}")
print(f"  intercept: {intercept:.4f}")
print(f"  r-value:   {r_val:.4f}")
print(f"  p-value:   {p_val:.6f}")

# --------------------------------------------------------
# PLOT
# --------------------------------------------------------

plt.scatter(X, Y)
plt.plot(X, slope * X + intercept)  # regression line

for i, subj in enumerate(subjects):
    plt.annotate(subj, (X[i], Y[i]))

plt.title("Lemma Variety vs. Accuracy Change")
plt.xlabel("Lemma Variety (unique lemmas)")
plt.ylabel("Avg Accuracy Change")
plt.grid(True)
plt.show()
