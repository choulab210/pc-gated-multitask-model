import numpy as np

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)


# ================================================================
# Simple adsorption test
# ================================================================

y_true_presence = np.array([
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0],
])

y_prob_presence = np.array([
    [0.9, 0.1],
    [0.8, 0.9],
    [0.2, 0.8],
    [0.1, 0.2],
])

presence_results = adsorption_metrics(
    y_true_presence,
    y_prob_presence,
)

print("Adsorption metrics:")
print(presence_results)


# ================================================================
# Simple abundance test
# ================================================================

y_true_abundance = np.array([
    [0.7, 0.3],
    [0.2, 0.8],
    [0.5, 0.5],
    [0.9, 0.1],
])

y_pred_abundance = np.array([
    [0.65, 0.35],
    [0.25, 0.75],
    [0.45, 0.55],
    [0.85, 0.15],
])

abundance_results = abundance_metrics(
    y_true_abundance,
    y_pred_abundance,
)

print()
print("Abundance metrics:")
print(abundance_results)