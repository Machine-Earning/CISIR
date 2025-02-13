import os
from modules.reweighting.ImportanceWeighting import LinearImportance as linear
from modules.reweighting.ImportanceWeighting import CosineImportance as cosine
from modules.reweighting.ImportanceWeighting import QUCImportance as quc
from modules.reweighting.ImportanceWeighting import ReciprocalImportance as reciprocal

from modules.shared.globals import *
from modules.training.ts_modeling import build_dataset, plot_importance


def main():
    # Build the training dataset
    print("Building the training dataset. Please wait...")

    X_train, y_train, _, _ = build_dataset(
        DS_PATH + '/training',
        inputs_to_use=INPUTS_TO_USE[0],
        add_slope=ADD_SLOPE[0],
        outputs_to_use=OUTPUTS_TO_USE,
        cme_speed_threshold=CME_SPEED_THRESHOLD[0]
    )

    print("Training dataset built successfully.")

    # Compute delta and prepare reweights
    delta_train = y_train[:, 0]
    alpha_mse = 0.4
    bandwidth = BANDWIDTH

    # Calculate weights for each importance type
    linear_weights = linear(
        X_train,
        delta_train,
        alpha=alpha_mse,
        bandwidth=bandwidth
    ).label_importance_map

    cosine_weights = cosine(
        X_train,
        delta_train,
        alpha=alpha_mse,
        bandwidth=bandwidth
    ).label_importance_map

    quc_weights = quc(
        X_train,
        delta_train,
        alpha=alpha_mse,
        bandwidth=bandwidth
    ).label_importance_map

    reciprocal_weights = reciprocal(
        X_train,
        delta_train,
        alpha=alpha_mse,
        bandwidth=bandwidth
    ).label_importance_map

    print("Weights computed successfully.")

    # Plot importance for each type
    plot_importance(linear_weights, tag=f"alpha_{alpha_mse}_linear_importance")
    plot_importance(cosine_weights, tag=f"alpha_{alpha_mse}_cosine_importance") 
    plot_importance(quc_weights, tag=f"alpha_{alpha_mse}_quc_importance")
    plot_importance(reciprocal_weights, tag=f"alpha_{alpha_mse}_reciprocal_importance")

    print("Done.")


if __name__ == '__main__':
    main()
