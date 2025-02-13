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
    alpha_values = [0.03, 0.1, 0.4, 1]  # List of alpha values to try
    bandwidth = BANDWIDTH

    # Create directories if they don't exist
    for dir_name in ['linear', 'cosine', 'quc', 'reciprocal']:
        os.makedirs(f'plots/{dir_name}', exist_ok=True)

    # Loop through each alpha value
    for alpha_mse in alpha_values:
        print(f"Computing weights for alpha = {alpha_mse}")
        
        # Calculate weights for each importance type
        # linear_weights = linear(
        #     X_train,
        #     delta_train,
        #     alpha=alpha_mse,
        #     bandwidth=bandwidth
        # ).label_importance_map

        # cosine_weights = cosine(
        #     X_train,
        #     delta_train,
        #     alpha=alpha_mse,
        #     bandwidth=bandwidth
        # ).label_importance_map

        # quc_weights = quc(
        #     X_train,
        #     delta_train,
        #     alpha=alpha_mse,
        #     bandwidth=bandwidth
        # ).label_importance_map

        reciprocal_weights = reciprocal(
            X_train,
            delta_train,
            alpha=alpha_mse,
            bandwidth=bandwidth
        ).label_importance_map

        print(f"Weights computed successfully for alpha = {alpha_mse}")

        # Plot importance for each type
        # plot_importance(linear_weights, tag=f"alpha_{alpha_mse}_linear_importance", 
        #                save_path=f"plots/linear/alpha_{alpha_mse}_linear_importance.png")
        # plot_importance(cosine_weights, tag=f"alpha_{alpha_mse}_cosine_importance",
        #                save_path=f"plots/cosine/alpha_{alpha_mse}_cosine_importance.png")
        # plot_importance(quc_weights, tag=f"alpha_{alpha_mse}_quc_importance",
        #                save_path=f"plots/quc/alpha_{alpha_mse}_quc_importance.png")
        plot_importance(reciprocal_weights, tag=f"alpha_{alpha_mse}_reciprocal_importance",
                       save_path=f"plots/reciprocal_smoothed/alpha_{alpha_mse}_reciprocal_importance.png")

    print("Done.")


if __name__ == '__main__':
    main()
