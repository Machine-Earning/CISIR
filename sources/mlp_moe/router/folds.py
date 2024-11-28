import os
from datetime import datetime

import numpy as np
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
from modules.reweighting.exDenseReweightsD import exDenseReweightsD
from modules.shared.globals import *
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    stratified_batch_dataset,
    set_seed,
    cmse,
    filter_ds,
    create_mlp,
    plot_error_hist,
    load_stratified_folds,
)


def main():
    """
    Main function to run the Router model
    :return:
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS:
            for rho in RHO:  # SAM_RHOS:
                # PARAMS
                inputs_to_use = INPUTS_TO_USE[0]  # Use first input configuration
                outputs_to_use = OUTPUTS_TO_USE
                lambda_factor = LAMBDA_FACTOR  # lambda for the loss
                add_slope = ADD_SLOPE[0]  # Use first add_slope value
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]  # Use first threshold value
                
                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'router_amse{alpha_mse:.2f}_v8_updated'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = PATIENCE  # higher patience
                learning_rate = START_LR  # starting learning rate
                asym_type = ASYM_TYPE

                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor=LR_CB_MONITOR,
                    factor=LR_CB_FACTOR,
                    patience=LR_CB_PATIENCE,
                    verbose=VERBOSE,
                    min_delta=LR_CB_MIN_DELTA,
                    min_lr=LR_CB_MIN_LR)

                weight_decay = WEIGHT_DECAY  # 1e-5 # higher weight decay
                momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                batch_size = BATCH_SIZE  # higher batch size
                epochs = EPOCHS  # higher epochs
                hiddens = MLP_HIDDENS  # hidden layers

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = ROUTER_OUTPUT_DIM  # 3 classes for routing
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                residual = RESIDUAL
                skip_repr = SKIP_REPR
                skipped_layers = SKIPPED_LAYERS
                N = N_FILTERED  # number of samples to keep outside the threshold
                lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                mae_plus_threshold = MAE_PLUS_THRESHOLD
                smoothing_method = SMOOTHING_METHOD
                window_size = WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = VAL_WINDOW_SIZE  # allows margin of error of 10 epochs

                # Initialize wandb
                wandb.init(project="Jan-Report", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    'min_lr': LR_CB_MIN_LR,
                    "weight_decay": weight_decay,
                    "momentum_beta1": momentum_beta1,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    # hidden in a more readable format  (wandb does not support lists)
                    "hiddens": hiddens_str,
                    "loss": 'categorical_crossentropy',
                    "lambda": lambda_factor,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "alphaV_mse": alphaV_mse,
                    "alpha_pcc": alpha_pcc,
                    "alphaV_pcc": alphaV_pcc,
                    "bandwidth": bandwidth,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "activation": 'LeakyReLU',
                    "norm": norm,
                    'optimizer': 'adamw',
                    'output_dim': output_dim,
                    'architecture': 'router',
                    'cme_speed_threshold': cme_speed_threshold,
                    'residual': residual,
                    'ds_version': DS_VERSION,
                    'mae_plus_th': mae_plus_threshold,
                    'sam_rho': rho,
                    'smoothing_method': smoothing_method,
                    'window_size': window_size,
                    'val_window_size': val_window_size,
                    'skip_repr': skip_repr,
                    'asym_type': asym_type
                })

                # set the root directory
                root_dir = DS_PATH
                # build the dataset
                X_train, y_train, logI_train, logI_prev_train = build_dataset(
                    root_dir + '/training',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold,
                    shuffle_data=True)

                # Convert y_train to 3 classes based on thresholds
                y_train_classes = np.zeros((y_train.shape[0], 3))
                y_train_classes[y_train[:,0] >= upper_threshold, 2] = 1  # High class
                y_train_classes[(y_train[:,0] > lower_threshold) & (y_train[:,0] < upper_threshold), 1] = 1  # Mid class  
                y_train_classes[y_train[:,0] <= lower_threshold, 0] = 1  # Low class

                # print the training set shapes
                print(f'X_train.shape: {X_train.shape}, y_train_classes.shape: {y_train_classes.shape}')

                # Build test set
                X_test, y_test, logI_test, logI_prev_test = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)

                # Convert y_test to 3 classes based on thresholds
                y_test_classes = np.zeros((y_test.shape[0], 3))
                y_test_classes[y_test[:,0] >= upper_threshold, 2] = 1  # High class
                y_test_classes[(y_test[:,0] > lower_threshold) & (y_test[:,0] < upper_threshold), 1] = 1  # Mid class
                y_test_classes[y_test[:,0] <= lower_threshold, 0] = 1  # Low class

                print(f'X_test.shape: {X_test.shape}, y_test_classes.shape: {y_test_classes.shape}')

                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                # Create router model
                router_model = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    embed_dim=embed_dim,
                    output_dim=output_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho,
                    output_activation='softmax'  # Use softmax for multi-class classification
                )

                router_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Train the router model
                history = router_model.fit(
                    X_train, y_train_classes,
                    validation_data=(X_test, y_test_classes),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Save the final model
                router_model.save_weights(f"router_model_weights_{experiment_name}.h5")
                print(f"Model weights saved in router_model_weights_{experiment_name}.h5")

                # Evaluate accuracy
                train_accuracy = router_model.evaluate(X_train, y_train_classes)[1]
                test_accuracy = router_model.evaluate(X_test, y_test_classes)[1]

                print(f'Train accuracy: {train_accuracy}')
                print(f'Test accuracy: {test_accuracy}')

                wandb.log({
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy
                })

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
