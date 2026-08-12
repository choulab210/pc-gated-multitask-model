from pcmodel.data import (
    load_data,
    split_abundance_data,
    select_protein_panel,
    build_targets,
)


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

features, abundance = load_data(
    "data/Data_1.csv",
    "data/Data_2.csv",
)


# ---------------------------------------------------------
# Split development and test samples
# ---------------------------------------------------------

train_df, test_df = split_abundance_data(
    abundance
)


# ---------------------------------------------------------
# Select protein panel using development data only
# ---------------------------------------------------------

panel_result = select_protein_panel(
    train_df
)

panel = panel_result.panel


# ---------------------------------------------------------
# Build targets
# ---------------------------------------------------------

(
    Y_presence_train,
    Y_abundance_train,
    presence_cols,
    abundance_cols,
) = build_targets(
    train_df,
    panel,
)


# ---------------------------------------------------------
# Print checks
# ---------------------------------------------------------

print("Feature dataset shape:", features.shape)
print("Abundance dataset shape:", abundance.shape)

print("Development samples:", len(train_df))
print("Test samples:", len(test_df))

print("Selected individual proteins:", len(panel))

print(
    "Presence target shape:",
    Y_presence_train.shape,
)

print(
    "Abundance target shape:",
    Y_abundance_train.shape,
)

print(
    "Presence output columns:",
    len(presence_cols),
)

print(
    "Abundance output columns:",
    len(abundance_cols),
)

print(
    "Eligible abundance fraction of all proteins:",
    panel_result.eligible_abundance_fraction_of_all,
)