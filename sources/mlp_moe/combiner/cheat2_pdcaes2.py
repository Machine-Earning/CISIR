from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report, accuracy_score
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
    set_seed,
    create_mlp,
    convert_to_onehot_cls,
    plot_confusion_matrix,
    create_metrics_table,
    stratified_batch_dataset_cls,
    filter_ds,
    plot_posteriors,
    pn_nz_loss,
    load_partial_weights_from_path
)


def main():
    """
    Main function to run the Combiner model
    :return:
    """

    # Path to pre-trained model weights
    pretrained_weights = COMBINER_PDCAE_S2
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for alpha_ce, alphaV_ce, alpha_pcc, alphaV_pcc in REWEIGHTS_MOE_C:
            for rho in RHO_MOE_C:  # SAM_RHOS:
                # PARAMS
                inputs_to_use = INPUTS_TO_USE[0]  # Use first input configuration
                outputs_to_use = OUTPUTS_TO_USE
                add_slope = ADD_SLOPE[0]  # Use first add_slope value
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]  # Use first threshold value
                lambda_pn = 0.0  # LAMBDA_PN_CCE
                lambda_nz = 0.0  # LAMBDA_NZ_CCE
                lambda_ce = 1.0 


                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'mlp2pdcaes2_ace{alpha_ce:.2f}_combiner_lpn{lambda_pn:.2f}_lnz{lambda_nz:.2f}_lce{lambda_ce:.2f}'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = PATIENCE_MOE_C  # higher patience
                learning_rate = START_LR_MOE_C # starting learning rate
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
                proj_hiddens = PROJ_HIDDENS

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = COMBINER_OUTPUT_DIM  # 3 classes for routing
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
                pretraining = True

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
                    "loss": 'pn_nz_loss',
                    "lambda_pn": lambda_pn,
                    "lambda_nz": lambda_nz,
                    "lambda_ce": lambda_ce,
                    "seed": seed,
                    "alpha_ce": alpha_ce,
                    "alphaV_ce": alphaV_ce,
                    "alpha_pcc": alpha_pcc,
                    "alphaV_pcc": alphaV_pcc,
                    "bandwidth": bandwidth,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "activation": 'LeakyReLU',
                    "norm": norm,
                    'optimizer': 'adamw',
                    'output_dim': output_dim,
                    'architecture': 'combiner',
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
                    'cvrg_metric': CVRG_METRIC,
                    'pretraining': pretraining,
                    'pretrained_weights': pretrained_weights,
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
                ce_train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_ce, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                pcc_train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_pcc, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                print(f'training set rebalanced.')

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
                ce_test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_ce, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                pcc_test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_pcc, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict

                # filtering training and test sets for additional results
                X_train_filtered, y_train_filtered = filter_ds(
                    X_train, y_train,
                    low_threshold=LOWER_THRESHOLD,  # std threshold for evals
                    high_threshold=UPPER_THRESHOLD,  # std threshold for evals
                    N=N, seed=seed)
                X_test_filtered, y_test_filtered = filter_ds(
                    X_test, y_test,
                    low_threshold=LOWER_THRESHOLD,  # std threshold for evals
                    high_threshold=UPPER_THRESHOLD,  # std threshold for evals
                    N=N, seed=seed)

                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                old_model_params = {
                    'input_dim': n_features,
                    'hiddens': hiddens,
                    'output_dim': 1,  # original output dimension
                    'pretraining': pretraining,
                    'embed_dim': embed_dim,
                    'dropout': dropout,
                    'activation': activation,
                    'norm': norm,
                    'skip_repr': skip_repr,
                    'skipped_layers': skipped_layers,
                    'sam_rho': rho,
                    'proj_hiddens': proj_hiddens
                }

                # Create initial model for finding optimal epochs
                initial_model = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    embed_dim=embed_dim,
                    output_dim=output_dim,
                    pretraining=pretraining,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho,
                    output_activation='softmax'
                )

                # Load pre-trained weights if available
                if pretrained_weights is not None:
                    print(f"Loading pre-trained weights from {pretrained_weights}")
                    load_partial_weights_from_path(
                        pretrained_weights_path=PRE_WEIGHT_PATH,
                        new_model=initial_model,
                        old_model_params=old_model_params,
                        skip_layers=["forecast_head"],  # skip final layer if output_dim differs
                        proj_neck=True,
                        no_head=False
                    )

                # summary of the model
                initial_model.summary()

                initial_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={
                        'forecast_head': lambda y_true, y_pred: pn_nz_loss(
                            y_true, y_pred,
                            phase_manager=pm,
                            loss_weights={'ce': lambda_ce, 'pn': lambda_pn, 'nz': lambda_nz},
                            train_ce_weight_dict=ce_train_weights_dict,
                            val_ce_weight_dict=ce_test_weights_dict,
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            val_pcc_weight_dict=pcc_test_weights_dict
                        )
                    }
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

                # Create stratified dataset for training
                train_dataset, steps_per_epoch = stratified_batch_dataset_cls(
                    X_train,
                    y_train_classes,
                    delta_train,
                    batch_size
                )

                # Map the dataset to include the 'forecast_head' key
                train_dataset = train_dataset.map(lambda x, y: (x, {'forecast_head': y}))

                # Prepare validation data without batching and concatenate y_test_classes and delta_test
                val_data = (
                X_test, {'forecast_head': tf.concat([y_test_classes, tf.expand_dims(delta_test, -1)], axis=1)})

                # Train initial model to find optimal epochs
                history = initial_model.fit(
                    train_dataset,
                    validation_data=val_data,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[
                        reduce_lr_on_plateau,
                        early_stopping,
                        WandbCallback(save_model=False),
                        IsTraining(pm)

                    ],
                    verbose=VERBOSE
                )

                # Find optimal epoch using validation loss
                val_losses = history.history['val_forecast_head_loss']
                optimal_epochs = find_optimal_epoch_by_smoothing(
                    val_losses,
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')

                print(f'optimal_epochs: {optimal_epochs}')
                wandb.log({'optimal_epochs': optimal_epochs})

                # Create final combiner model
                combiner_model = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    embed_dim=embed_dim,
                    output_dim=output_dim,
                    pretraining=pretraining,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho,
                    output_activation='softmax'
                )
                # summary of the model
                combiner_model.summary()

                combiner_model.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={
                        'forecast_head': lambda y_true, y_pred: pn_nz_loss(
                            y_true, y_pred,
                            phase_manager=pm,
                            loss_weights={'ce': lambda_ce, 'pn': lambda_pn, 'nz': lambda_nz},
                            train_ce_weight_dict=ce_train_weights_dict,
                            val_ce_weight_dict=ce_test_weights_dict,
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            val_pcc_weight_dict=pcc_test_weights_dict
                        )
                    }
                )

                # Load pre-trained weights if available
                if pretrained_weights is not None:
                    print(f"Loading pre-trained weights from {pretrained_weights}")
                    load_partial_weights_from_path(
                        pretrained_weights_path=PRE_WEIGHT_PATH,
                        new_model=combiner_model,
                        old_model_params=old_model_params,
                        skip_layers=["forecast_head"],  # skip final layer if output_dim differs
                        proj_neck=True,
                        no_head=False
                    )

                # Train the combiner model for optimal epochs
                history = combiner_model.fit(
                    train_dataset,
                    validation_data=val_data,
                    epochs=optimal_epochs,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Save the final model
                combiner_model.save_weights(f"final_combiner_model_weights_{experiment_name}.h5")
                print(f"Model weights saved in final_combiner_model_weights_{experiment_name}.h5")

                # Get predictions
                predictions = combiner_model.predict(X_train)
                y_train_pred = predictions[1]
                predictions = combiner_model.predict(X_test)
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
                )

                # Create and save test confusion matrix plot
                test_cm_fig = plot_confusion_matrix(
                    y_test_pred_classes,  # Switched order to have predicted on y-axis
                    y_test_true_classes,  # Switched order to have actual on x-axis
                    class_names=class_names,
                    title="Test Confusion Matrix",
                    xlabel="Actual",  # Added x-label for actual values
                    ylabel="Predicted",  # Added y-label for predicted values
                )

                # Calculate accuracies
                train_accuracy = accuracy_score(y_train_true_classes, y_train_pred_classes)
                test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)

                # Calculate class-specific accuracies
                train_plus_mask = y_train_true_classes == PLUS_INDEX  # plus class index
                train_zero_mask = y_train_true_classes == MID_INDEX  # zero class index
                train_minus_mask = y_train_true_classes == MINUS_INDEX  # minus class index

                test_plus_mask = y_test_true_classes == PLUS_INDEX  # plus class index
                test_zero_mask = y_test_true_classes == MID_INDEX  # zero class index
                test_minus_mask = y_test_true_classes == MINUS_INDEX  # minus class index

                train_plus_accuracy = accuracy_score(y_train_true_classes[train_plus_mask],
                                                     y_train_pred_classes[train_plus_mask])
                train_zero_accuracy = accuracy_score(y_train_true_classes[train_zero_mask],
                                                     y_train_pred_classes[train_zero_mask])
                train_minus_accuracy = accuracy_score(y_train_true_classes[train_minus_mask],
                                                      y_train_pred_classes[train_minus_mask])

                test_plus_accuracy = accuracy_score(y_test_true_classes[test_plus_mask],
                                                    y_test_pred_classes[test_plus_mask])
                test_zero_accuracy = accuracy_score(y_test_true_classes[test_zero_mask],
                                                    y_test_pred_classes[test_zero_mask])
                test_minus_accuracy = accuracy_score(y_test_true_classes[test_minus_mask],
                                                     y_test_pred_classes[test_minus_mask])

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

                # Plot posteriors
                train_posteriors_fig = plot_posteriors(y_train_pred, y_train, 
                                                       suptitle="Train Posterior Probabilities vs. Delta")
                test_posteriors_fig = plot_posteriors(y_test_pred, y_test, 
                                                      suptitle="Test Posterior Probabilities vs. Delta")

                # Update the wandb.log call to include the tables as images and class-specific accuracies
                wandb.log({
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train+_accuracy": train_plus_accuracy,
                    "train0_accuracy": train_zero_accuracy,
                    "train-_accuracy": train_minus_accuracy,
                    "test+_accuracy": test_plus_accuracy,
                    "test0_accuracy": test_zero_accuracy,
                    "test-_accuracy": test_minus_accuracy,
                    "train_confusion_matrix": wandb.Image(train_cm_fig),
                    "test_confusion_matrix": wandb.Image(test_cm_fig),
                    "train_classification_metrics": wandb.Image(train_metrics_fig),
                    "test_classification_metrics": wandb.Image(test_metrics_fig),
                    "train_posteriors": wandb.Image(train_posteriors_fig),
                    "test_posteriors": wandb.Image(test_posteriors_fig)
                })

                # Close all figures to free memory
                plt.close(train_cm_fig)
                plt.close(test_cm_fig)
                plt.close(train_metrics_fig)
                plt.close(test_metrics_fig)
                plt.close(train_posteriors_fig)
                plt.close(test_posteriors_fig)


                # Evaluate the model correlation with colored
                file_path = plot_repr_corr_dist(
                    combiner_model,
                    X_train_filtered, y_train_filtered,
                    title + "_training",
                    model_type='features_cls')
                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_corr_dist(
                    combiner_model,
                    X_test_filtered, y_test_filtered,
                    title + "_test",
                    model_type='features_cls')
                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Log t-SNE plot
                # Log the training t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    combiner_model,
                    X_train_filtered, y_train_filtered,
                    title, 'combiner_training',
                    model_type='features_cls',
                    save_tag=current_time, seed=seed)
                wandb.log({'combiner_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Log the testing t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    combiner_model,
                    X_test_filtered, y_test_filtered,
                    title, 'combiner_testing',
                    model_type='features_cls',
                    save_tag=current_time, seed=seed)
                wandb.log({'combiner_tsne_testing_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
