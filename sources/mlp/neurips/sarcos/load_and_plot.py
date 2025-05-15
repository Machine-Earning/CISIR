import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from typing import Tuple

from modules.shared.sarcos_globals import *
from modules.training.ts_modeling import (
    build_sarcos_ds,
    create_mlp,
    plot_sarcos_cisir
)

def main():
    """
    Load a pre-trained model and plot using plot_sarcos_cisir on test set
    """
    # Set a fixed seed for reproducibility
    seed = 42
    # Model configuration parameters
    MODEL_PATH = "final_model_weights_mlp_amse1.00_apcc0.00_denseloss_20250411-092807_reg.h5"
    hiddens = MLP_HIDDENS
    embed_dim = EMBED_DIM
    output_dim = OUTPUT_DIM
    dropout = DROPOUT
    activation = ACTIVATION
    norm = NORM
    skip_repr = SKIP_REPR
    skipped_layers = SKIPPED_LAYERS
    rho = 0.0  # No SAM for inference
    weight_decay = WEIGHT_DECAY
    pretraining = False
    
    # Set the root directory
    root_dir = DS_PATH
    
    # Load test data
    X_test, y_test = build_sarcos_ds(
        root_dir + '/sarcos_inv_testing.csv',
        shuffle_data=False,
        random_state=seed
    )
    
    # Print the test set shapes
    print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
    
    # Get the number of input features
    n_features = X_test.shape[1]
    print(f'n_features: {n_features}')
    
    # Create the model with the same architecture as during training
    model = create_mlp(
        input_dim=n_features,
        hiddens=hiddens,
        embed_dim=embed_dim,
        output_dim=output_dim,
        dropout=dropout,
        activation=activation,
        norm=norm,
        skip_repr=skip_repr,
        skipped_layers=skipped_layers,
        pretraining=pretraining,
        sam_rho=rho,
        weight_decay=weight_decay
    )
    
    # Compile the model (simple compilation for inference)
    model.compile(
        optimizer=Adam(),
        loss={'forecast_head': 'mse'}
    )
    
    # Load weights from file
    # You should specify the path to your model weights file
    weights_file = MODEL_PATH  # Update this path
    try:
        model.load_weights(weights_file)
        print(f"Successfully loaded weights from {weights_file}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Plot using plot_sarcos_cisir
    title = "(c) DenseLoss"
    # title = "(d) Recip+wPCC+SSB"
    # title = "(e) CISIR"
    lower_threshold = LOWER_THRESHOLD
    upper_threshold = UPPER_THRESHOLD
    
    filename = plot_sarcos_cisir(
        model,
        X_test, 
        y_test,
        title=title,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        output_dir="./plots",
        filename_prefix="sarcos_test",
        use_dict=False,
        figsize=(12, 7)
    )
    
    print(f"Plot saved to: {filename}")

if __name__ == '__main__':
    main()
