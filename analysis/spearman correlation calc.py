from scipy.stats import spearmanr

# order of language values: ar,de,en,fr,hi,it
x = [16.31205674,12.02185792,5.159527859,7.729708824,5.977165883,0.005494203615]
y = [-0.662, -0.05, 0.702, -0.268, -0.548, 0.82]

rho, p_value = spearmanr(x, y)

print(f"Spearman correlation: {rho:.4f}")
print(f"p-value: {p_value:.4g}")



