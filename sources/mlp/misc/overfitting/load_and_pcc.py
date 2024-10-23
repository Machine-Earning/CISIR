import os

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import tensorflow as tf
import wandb
import numpy as np

from modules.evaluate.utils import (
    evaluate_pcc_repr,
)
from modules.training.cme_modeling import ModelBuilder
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp, filter_ds)
from modules.shared.globals import *

mb = ModelBuilder()

# Define the lookup dictionary
weight_paths = {
    # No inj
    'noinj': '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240627-012209MLP_e0_5_e1_8_p_slopeFalse_PDS_bs4096_alpha0.20_CME0_features_noinj.h5',
    # inj all
    'injall': '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240627-030006MLP_e0_5_e1_8_p_slopeFalse_PDSinj_bs4096_alpha0.20_CME0_features_all.h5',
    # inj min
    'injmin': '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240626-201049MLP_e0_5_e1_8_p_slopeFalse_PDSinj_bs4096_alpha1.00_CME0_features_min.h5',
}


def main():
    """
    Main function to run the E-MLP model and evaluate PCC for each weight configuration
    """
    # Set up constant parameters
    inputs_to_use = INPUTS_TO_USE[0]
    add_slope = ADD_SLOPE[0]  # Adjust as needed
    outputs_to_use = OUTPUTS_TO_USE
    cme_speed_threshold = CME_SPEED_THRESHOLD[0]  # Adjust as needed

    # Other constant parameters...
    seed = SEEDS[0]
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Build datasets
    root_dir = DS_PATH
    X_test, y_test = build_dataset(
        root_dir + '/testing',
        inputs_to_use=inputs_to_use,
        add_slope=add_slope,
        outputs_to_use=outputs_to_use,
        cme_speed_threshold=cme_speed_threshold)

    X_test_filtered, y_test_filtered = filter_ds(
        X_test, y_test,
        low_threshold=LOWER_THRESHOLD,
        high_threshold=UPPER_THRESHOLD,
        N=N_FILTERED, seed=seed)

    n_features = X_test.shape[1]

    # Create model architecture (without loading weights)
    model = create_mlp(
        input_dim=n_features,
        hiddens=MLP_HIDDENS,
        output_dim=0,
        pds=True,
        repr_dim=REPR_DIM,
        dropout=DROPOUT,
        activation=ACTIVATION,
        norm=NORM,
        residual=RESIDUAL,
        skipped_layers=SKIPPED_LAYERS
    )

    # Initialize wandb
    wandb.init(project="nasa-ts-delta-plot", name="weight_comparison_experiment")

    # Loop through weight paths
    for key, weight_path in weight_paths.items():
        print(f"\nEvaluating weights for: {key}")
        print(f"Weight path: {weight_path}")

        # Load weights
        model.load_weights(weight_path)
        print(f"Weights loaded successfully from: {weight_path}")

        # Evaluate PCC
        pcc = evaluate_pcc_repr(model, X_test_filtered, y_test_filtered)
        print(f"PCC for {key}: {pcc}")

        # Evaluate conditional PCC (i >= 0.5)
        pcc_cond = evaluate_pcc_repr(model, X_test_filtered, y_test_filtered, i_above_threshold=0.5)
        print(f"Conditional PCC (i >= 0.5) for {key}: {pcc_cond}")

        # Log to wandb
        wandb.log({
            f"pcc_{key}": pcc,
            f"pcc_cond_{key}": pcc_cond
        })

    # Finish the wandb run
    wandb.finish()


if __name__ == '__main__':
    main()
