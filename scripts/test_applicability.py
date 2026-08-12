from pcmodel.data import prepare_model_data

from pcmodel.applicability import (
    fit_applicability_domain,
)


data = prepare_model_data(
    "data/Data_1.csv",
    "data/Data_2.csv",
)


# ================================================================
# Fit AD using model-development samples only
# ================================================================

ad_model = fit_applicability_domain(
    data.X_train,
    n_neighbors=5,
    threshold_percentile=85.0,
)


# ================================================================
# Apply to held-out test samples
# ================================================================

result = ad_model.evaluate(
    data.X_test
)


print("AD threshold:")
print(result.threshold)

print()

print("Test samples:")
print(len(result.in_domain))

print(
    "In-AD:",
    int(
        result.in_domain.sum()
    )
)

print(
    "Out-of-AD:",
    int(
        (~result.in_domain).sum()
    )
)

print()

print("AD model summary:")
print(
    ad_model.summary()
)