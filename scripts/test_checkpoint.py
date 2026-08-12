"""
Test loading the saved two-head model checkpoint.
"""

from pathlib import Path

import torch

from pcmodel.models import TwoHead


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "twohead_model_checkpoint.pt"
)


# ================================================================
# Load checkpoint
# ================================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True,
)

print("Checkpoint loaded successfully.")

print()
print("Saved information:")
print("Input dimensions:", checkpoint["input_dim"])
print("Outputs:", checkpoint["n_outputs"])
print("Selected proteins:", len(checkpoint["panel"]))
print("Best epoch:", checkpoint["best_epoch"])
print("Best score:", checkpoint["best_score"])


# ================================================================
# Reconstruct model
# ================================================================

config = checkpoint["training_config"]

model = TwoHead(
    in_dim=checkpoint["input_dim"],
    hidden=list(config["hidden"]),
    k=checkpoint["n_outputs"],
    dropout=config["dropout"],
    alpha_gate=config["alpha_gate"],
    temp_init=config["temp_init"],
    stopgrad_gate=True,
)


# ================================================================
# Load model weights
# ================================================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print()
print("Model weights loaded successfully.")
print("Checkpoint test successful.")