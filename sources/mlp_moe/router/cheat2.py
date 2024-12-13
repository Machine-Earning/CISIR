from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.shared.globals import *
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_dataset,
    set_seed,
    create_mlp,
    convert_to_onehot_cls,
    focal_loss,
    plot_confusion_matrix
)


def main():
    """
    Main function to run the Router model
    :return:
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS_MOE:
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
                title = f'mlp2_amse{alpha_mse:.2f}_router'
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
                lower_threshold = LOWER_THRESHOLD_MOE  # lower threshold for the delta_p
                upper_threshold = UPPER_THRESHOLD_MOE  # upper threshold for the delta_p
                mae_plus_threshold = MAE_PLUS_THRESHOLD
                smoothing_method = SMOOTHING_METHOD
                window_size = WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = VAL_WINDOW_SIZE  # allows margin of error of 10 epochs

                # Initialize wandb
                wandb.init(project="Jan-moe-Report", name=experiment_name, config={
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
                    "loss": 'focal_loss',
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
                    'asym_type': asym_type,
                    'upper_threshold': upper_threshold,
                    'lower_threshold': lower_threshold,
                    'cvrg_metric': CVRG_METRIC
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
                y_train_classes = convert_to_onehot_cls(y_train, lower_threshold, upper_threshold)

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
                y_test_classes = convert_to_onehot_cls(y_test, lower_threshold, upper_threshold)

                print(f'X_test.shape: {X_test.shape}, y_test_classes.shape: {y_test_classes.shape}')

                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                # Create initial model for finding optimal epochs
                initial_model = create_mlp(
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
                    output_activation='softmax'
                )

                initial_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={'forecast_head': focal_loss(gamma=2.0, alpha=0.25)},
                    metrics={'forecast_head': 'accuracy'}
                )

                # Create early stopping callback
                early_stopping = SmoothEarlyStopping(
                    monitor=CVRG_METRIC,
                    min_delta=CVRG_MIN_DELTA,
                    patience=patience,
                    verbose=VERBOSE,
                    mode='min',
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': window_size}
                )

                # Train initial model to find optimal epochs
                history = initial_model.fit(
                    X_train, {'forecast_head': y_train_classes},
                    validation_data=(X_test, {'forecast_head': y_test_classes}),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        early_stopping,
                        WandbCallback(save_model=False),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Find optimal epoch using test loss
                val_losses = history.history['val_forecast_head_loss']
                optimal_epochs = find_optimal_epoch_by_smoothing(
                    val_losses,
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')

                print(f'optimal_epochs: {optimal_epochs}')
                wandb.log({'optimal_epochs': optimal_epochs})

                # Create final router model
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
                    output_activation='softmax'
                )

                router_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={'forecast_head': focal_loss(gamma=2.0, alpha=0.25)},
                    metrics={'forecast_head': 'accuracy'}
                )

                # Train the router model for optimal epochs
                history = router_model.fit(
                    X_train, {'forecast_head': y_train_classes},
                    validation_data=(X_test, {'forecast_head': y_test_classes}),
                    epochs=optimal_epochs,
                    batch_size=batch_size,
                    callbacks=[
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Save the final model
                router_model.save_weights(f"router_model_weights_{experiment_name}.h5")
                print(f"Model weights saved in router_model_weights_{experiment_name}.h5")

                # Get predictions
                predictions = router_model.predict(X_train)
                y_train_pred = predictions[1]
                predictions = router_model.predict(X_test)
                y_test_pred = predictions[1]

                # Convert predictions to class labels
                y_train_pred_classes = np.argmax(y_train_pred, axis=1)
                y_test_pred_classes = np.argmax(y_test_pred, axis=1)
                y_train_true_classes = np.argmax(y_train_classes, axis=1)
                y_test_true_classes = np.argmax(y_test_classes, axis=1)

                # Calculate confusion matrices and create plots
                class_names = ['Low', 'Mid', 'High']

                # Create and save train confusion matrix plot
                train_cm_fig = plot_confusion_matrix(
                    y_train_true_classes,
                    y_train_pred_classes,
                    class_names=class_names,
                    title="Training Confusion Matrix"
                )

                # Create and save test confusion matrix plot
                test_cm_fig = plot_confusion_matrix(
                    y_test_true_classes,
                    y_test_pred_classes,
                    class_names=class_names,
                    title="Test Confusion Matrix"
                )

                # Calculate accuracies
                train_metrics = router_model.evaluate(X_train, {'forecast_head': y_train_classes})
                test_metrics = router_model.evaluate(X_test, {'forecast_head': y_test_classes})
                train_accuracy = train_metrics[1]  # Assuming accuracy is the second metric
                test_accuracy = test_metrics[1]  # Assuming accuracy is the second metric

                # Get detailed classification reports
                train_report = classification_report(y_train_true_classes, y_train_pred_classes)
                test_report = classification_report(y_test_true_classes, y_test_pred_classes)

                print("\nTraining Results:")
                print(f'Accuracy: {train_accuracy}')
                print("\nClassification Report:")
                print(train_report)

                print("\nTest Results:")
                print(f'Accuracy: {test_accuracy}')
                print("\nClassification Report:")
                print(test_report)

                # Log metrics and plots to wandb
                wandb.log({
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_confusion_matrix": wandb.Image(train_cm_fig),
                    "test_confusion_matrix": wandb.Image(test_cm_fig)
                })

                # Close the figures to free memory
                plt.close(train_cm_fig)
                plt.close(test_cm_fig)

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
