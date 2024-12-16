from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.reweighting.exDenseReweightsD import exDenseReweightsD
from modules.shared.globals import *
from modules.training.phase_manager import create_weight_tensor_fast
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_dataset,
    set_seed,
    create_mlp,
    convert_to_onehot_cls,
    plot_confusion_matrix
)


# Create classification report tables for wandb
def create_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, set_name: str) -> plt.Figure:
    """
    Creates a matplotlib figure containing classification metrics for model evaluation.

    Args:
        y_true (np.ndarray): Ground truth labels (class indices)
        y_pred (np.ndarray): Predicted labels (class indices) 
        set_name (str): Name of the dataset split ('Train' or 'Test')

    Returns:
        plt.Figure: Figure containing accuracy, per-class accuracy, precision, recall and F1 scores
    """
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate per-class accuracy
    class_accuracies = []
    for i in range(3):  # 3 classes
        mask = y_true == i
        class_accuracies.append(accuracy_score(y_true[mask], y_pred[mask]))
    
    # Calculate other metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Define table data
    table_data = [
        ["Overall Accuracy", f"{accuracy:.3f}", "", "", ""],
        ["Class", "Accuracy", "Precision", "Recall", "F1-Score"],
        ["plus", f"{class_accuracies[2]:.3f}", f"{precision[2]:.3f}", f"{recall[2]:.3f}", f"{f1[2]:.3f}"],
        ["zero", f"{class_accuracies[1]:.3f}", f"{precision[1]:.3f}", f"{recall[1]:.3f}", f"{f1[1]:.3f}"],
        ["minus", f"{class_accuracies[0]:.3f}", f"{precision[0]:.3f}", f"{recall[0]:.3f}", f"{f1[0]:.3f}"]
    ]
    
    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Set title
    plt.title(f"{set_name} Classification Metrics", pad=20)
    
    return fig

def main():
    """
    Main function to run the Router model
    :return:
    """

    for seed in SEEDS:
        for alpha_ce, alphaV_ce in REWEIGHTS_MOE_R:
            for rho in RHO_MOE_R:  # SAM_RHOS:
                # PARAMS
                inputs_to_use = INPUTS_TO_USE[0]  # Use first input configuration
                outputs_to_use = OUTPUTS_TO_USE
                add_slope = ADD_SLOPE[0]  # Use first add_slope value
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]  # Use first threshold value

                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'mlp2_ace{alpha_ce:.2f}_router'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = PATIENCE_MOE  # higher patience
                learning_rate = START_LR  # starting learning rate
                asym_type = ASYM_TYPE_MOE

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
                wandb.init(project="Jan-moe-router-Report", name=experiment_name, config={
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
                    "loss": 'ce',
                    "seed": seed,
                    "alpha_ce": alpha_ce,
                    "alphaV_ce": alphaV_ce,
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
                X_train, y_train, _, _ = build_dataset(
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

                # Get sample weights for training set
                delta_train = y_train[:, 0]
                print(f'delta_train.shape: {delta_train.shape}')
                print(f'rebalancing the training set...')
                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_ce, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict

                # Build test set
                X_test, y_test, _, _ = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)

                # Convert y_test to 3 classes based on thresholds
                y_test_classes = convert_to_onehot_cls(y_test, lower_threshold, upper_threshold)

                print(f'X_test.shape: {X_test.shape}, y_test_classes.shape: {y_test_classes.shape}')

                # Get sample weights for test set
                delta_test = y_test[:, 0]
                print(f'delta_test.shape: {delta_test.shape}')
                print(f'rebalancing the test set...')
                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_ce, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict

                # Create weight tensors for train and test sets
                train_weights = create_weight_tensor_fast(y_train[:, 0], train_weights_dict)
                test_weights = create_weight_tensor_fast(y_test[:, 0], test_weights_dict)

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
                # summary of the model
                initial_model.summary()

                initial_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={'forecast_head': 'categorical_crossentropy'},
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
                    validation_data=(X_test, {'forecast_head': y_test_classes}, test_weights),
                    sample_weight=train_weights,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        early_stopping,
                        WandbCallback(save_model=False)
                    ],
                    verbose=VERBOSE
                )

                # Find optimal epoch using validation accuracy
                val_accuracies = history.history['val_forecast_head_accuracy']
                optimal_epochs = find_optimal_epoch_by_smoothing(
                    val_accuracies,
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='max')

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
                # summary of the model
                router_model.summary()

                router_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={'forecast_head': 'categorical_crossentropy'},
                    metrics={'forecast_head': 'accuracy'}
                )

                # Train the router model for optimal epochs
                history = router_model.fit(
                    X_train, {'forecast_head': y_train_classes},
                    validation_data=(X_test, {'forecast_head': y_test_classes}, test_weights),
                    sample_weight=train_weights,
                    epochs=optimal_epochs,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=WANDB_SAVE_MODEL)
                    ],
                    verbose=VERBOSE
                )

                # Save the final model
                router_model.save_weights(f"final_router_model_weights_{experiment_name}.h5")
                print(f"Model weights saved in final_router_model_weights_{experiment_name}.h5")

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
                class_names = ['minus', 'zero', 'plus']

                # Create and save train confusion matrix plot
                train_cm_fig = plot_confusion_matrix(
                    y_train_pred_classes,  # Switched order to have predicted on y-axis
                    y_train_true_classes,  # Switched order to have actual on x-axis
                    class_names=class_names,
                    title="Training Confusion Matrix",
                    xlabel="Actual",  # Added x-label for actual values
                    ylabel="Predicted",  # Added y-label for predicted values
                    xticklabels=class_names,  # Add class names on x-axis
                    yticklabels=class_names  # Add class names on y-axis
                )

                # Create and save test confusion matrix plot
                test_cm_fig = plot_confusion_matrix(
                    y_test_pred_classes,  # Switched order to have predicted on y-axis
                    y_test_true_classes,  # Switched order to have actual on x-axis
                    class_names=class_names,
                    title="Test Confusion Matrix",
                    xlabel="Actual",  # Added x-label for actual values
                    ylabel="Predicted",  # Added y-label for predicted values
                    xticklabels=class_names,  # Add class names on x-axis
                    yticklabels=class_names  # Add class names on y-axis
                )

                # Calculate accuracies
                train_accuracy = accuracy_score(y_train_true_classes, y_train_pred_classes)
                test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)

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

                # Create metric tables as figures
                train_metrics_fig = create_metrics_table(y_train_true_classes, y_train_pred_classes, "Train")
                test_metrics_fig = create_metrics_table(y_test_true_classes, y_test_pred_classes, "Test")

                # Update the wandb.log call to include the tables as images
                wandb.log({
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_confusion_matrix": wandb.Image(train_cm_fig),
                    "test_confusion_matrix": wandb.Image(test_cm_fig),
                    "train_classification_metrics": wandb.Image(train_metrics_fig),
                    "test_classification_metrics": wandb.Image(test_metrics_fig)
                })

                # Close all figures to free memory
                plt.close(train_cm_fig)
                plt.close(test_cm_fig)
                plt.close(train_metrics_fig)
                plt.close(test_metrics_fig)

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
