from pcmodel.data import prepare_model_data

from pcmodel.training import (
    TrainingConfig,
    train_with_early_stopping,
    predict_probabilities,
)

from pcmodel.metrics import (
    adsorption_metrics,
    abundance_metrics,
)


# ================================================================
# Prepare data
# ================================================================

data = prepare_model_data(
    "data/Data_1.csv",
    "data/Data_2.csv",
)


print("X train:", data.X_train.shape)
print("Presence:", data.Y_presence_train.shape)
print("Abundance:", data.Y_abundance_train.shape)


# ================================================================
# Short test configuration
# ================================================================

# Only 3 epochs for checking that the pipeline works.
config = TrainingConfig(
    max_epochs=3,
    patience=3,
)


# ================================================================
# Train
# ================================================================

result = train_with_early_stopping(
    data.X_train,
    data.Y_presence_train,
    data.Y_abundance_train,
    config=config,
)


# ================================================================
# Test-set prediction
# ================================================================

presence_prob, abundance_pred = (
    predict_probabilities(
        result.model,
        data.X_test,
    )
)


print()
print(
    "Presence prediction:",
    presence_prob.shape,
)

print(
    "Abundance prediction:",
    abundance_pred.shape,
)


# ================================================================
# Evaluate individual proteins only
# ================================================================

# OTHER is currently the final output, so exclude it for manuscript
# protein-level evaluation.

protein_indices = list(
    range(
        len(data.panel)
    )
)


presence_results = (
    adsorption_metrics(
        data.Y_presence_test,
        presence_prob,
        indices=protein_indices,
    )
)


abundance_results = (
    abundance_metrics(
        data.Y_abundance_test,
        abundance_pred,
        indices=protein_indices,
    )
)


print()
print("Adsorption:")
print(presence_results)

print()
print("Abundance:")
print(abundance_results)