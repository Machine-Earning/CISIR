import os
from datetime import datetime

import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback
from modules.training.smooth_early_stopping import SmoothEarlyStopping
from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
from modules.reweighting.exDenseReweightsD import exDenseReweightsD
from modules.shared.globals import *
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    stratified_batch_dataset,
    set_seed,
    cmse,
    filter_ds,
    create_gru,
    plot_error_hist,
    reshape_X
)


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    """
    Main function to run the training
    :return:
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS:
                    for rho in RHO:  # SAM_RHOS:
                        for add_slope in ADD_SLOPE:
                            # PARAMS
                            outputs_to_use = OUTPUTS_TO_USE
                            lambda_factor = LAMBDA_FACTOR  # lambda for the loss
                            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                            # Construct the title
                            title = f'gru_amse{alpha_mse:.2f}_cheat_NS'
                            # Replace any other characters that are not suitable for filenames (if any)
                            title = title.replace(' ', '_').replace(':', '_')
                            # Create a unique experiment name with a timestamp
                            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                            experiment_name = f'{title}_{current_time}'
                            # Set the early stopping patience and learning rate as variables
                            set_seed(seed)
                            patience = PATIENCE  # higher patience
                            learning_rate = START_LR  # starting learning rate

                            reduce_lr_on_plateau = ReduceLROnPlateau(
                                monitor=LR_CB_MONITOR,
                                factor=LR_CB_FACTOR,
                                patience=LR_CB_PATIENCE,
                                verbose=VERBOSE,
                                min_delta=LR_CB_MIN_DELTA,
                                min_lr=LR_CB_MIN_LR)

                            weight_decay = WEIGHT_DECAY  # higher weight decay
                            momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                            batch_size = BATCH_SIZE  # higher batch size
                            epochs = EPOCHS  # higher epochs
                            
                            # GRU specific parameters
                            gru_units = 30
                            gru_layers = 1
                            hiddens_str = f'{gru_units}units_{gru_layers}layers'
                            
                            bandwidth = BANDWIDTH
                            repr_dim = REPR_DIM
                            output_dim = len(outputs_to_use)
                            dropout = DROPOUT
                            activation = ACTIVATION
                            norm = NORM
                            cme_speed_threshold = cme_speed_threshold
                            residual = RESIDUAL
                            skipped_layers = SKIPPED_LAYERS
                            N = N_FILTERED  # number of samples to keep outside the threshold
                            lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                            upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                            mae_plus_threshold = MAE_PLUS_THRESHOLD
                            smoothing_method = SMOOTHING_METHOD
                            window_size = WINDOW_SIZE  # allows margin of error of 10 epochs

                            # Initialize wandb
                            wandb.init(project="Arch-test-gru", name=experiment_name, config={
                                "inputs_to_use": inputs_to_use,
                                "add_slope": add_slope,
                                "patience": patience,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "momentum_beta1": momentum_beta1,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                "hiddens": hiddens_str,
                                "loss": 'mse_pcc',
                                "lambda": lambda_factor,
                                "seed": seed,
                                "alpha_mse": alpha_mse,
                                "alphaV_mse": alphaV_mse,
                                "alpha_pcc": alpha_pcc,
                                "alphaV_pcc": alphaV_pcc,
                                "bandwidth": bandwidth,
                                "repr_dim": repr_dim,
                                "dropout": dropout,
                                "activation": 'LeakyReLU',
                                "norm": norm,
                                'optimizer': 'adamw',
                                'output_dim': output_dim,
                                'architecture': 'gru',
                                'cme_speed_threshold': cme_speed_threshold,
                                'residual': residual,
                                'ds_version': DS_VERSION,
                                'mae_plus_th': mae_plus_threshold,
                                'sam_rho': rho,
                            })

                            # set the root directory
                            root_dir = DS_PATH
                            # build the dataset
                            X_train, y_train = build_dataset(
                                root_dir + '/training',
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                cme_speed_threshold=cme_speed_threshold)

                            X_test, y_test = build_dataset(
                                root_dir + '/testing',
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                cme_speed_threshold=cme_speed_threshold)

                            # Filter datasets
                            X_train_filtered, y_train_filtered = filter_ds(X_train, y_train, N)
                            X_test_filtered, y_test_filtered = filter_ds(X_test, y_test, N)

                            # Create stratified batch datasets
                            train_ds = stratified_batch_dataset(
                                X_train, y_train,
                                batch_size=batch_size,
                                lower_threshold=lower_threshold,
                                upper_threshold=upper_threshold)

                            test_ds = stratified_batch_dataset(
                                X_test, y_test,
                                batch_size=batch_size,
                                lower_threshold=lower_threshold,
                                upper_threshold=upper_threshold)

                            # Calculate number of features
                            if add_slope:
                                n_features = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
                            else:
                                n_features = [25] * len(inputs_to_use)

                            # Create GRU model
                            model_sep = create_gru(
                                input_dims=n_features,
                                gru_units=gru_units,
                                gru_layers=gru_layers,
                                repr_dim=repr_dim,
                                output_dim=output_dim,
                                dropout=dropout,
                                activation=activation,
                                norm=norm
                            )

                            # Compile model
                            model_sep.compile(
                                optimizer=AdamW(
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    beta_1=momentum_beta1
                                ),
                                loss=cmse(lambda_factor, pm)
                            )

                            # Reshape inputs
                            X_train = reshape_X(X_train, n_features, inputs_to_use, add_slope, 'gru')
                            X_test = reshape_X(X_test, n_features, inputs_to_use, add_slope, 'gru')

                            # Create callbacks
                            smooth_early_stopping = SmoothEarlyStopping(
                                monitor='val_loss',
                                patience=patience,
                                smoothing_method=smoothing_method,
                                window_size=window_size,
                                verbose=VERBOSE,
                                restore_best_weights=True
                            )

                            callbacks = [
                                smooth_early_stopping,
                                reduce_lr_on_plateau,
                                WandbCallback(save_model=False)
                            ]

                            # Train model
                            history = model_sep.fit(
                                train_ds,
                                epochs=epochs,
                                validation_data=test_ds,
                                callbacks=callbacks,
                                verbose=VERBOSE
                            )

                            # Get final model
                            final_model_sep = model_sep

                            # Evaluate model
                            error_mae = evaluate_mae(final_model_sep, X_test, y_test)
                            print(f'mae error: {error_mae}')
                            wandb.log({"mae": error_mae})

                            error_mae_train = evaluate_mae(final_model_sep, X_train, y_train)
                            print(f'mae error train: {error_mae_train}')
                            wandb.log({"train_mae": error_mae_train})

                            error_pcc = evaluate_pcc(final_model_sep, X_test, y_test)
                            print(f'pcc error: {error_pcc}')
                            wandb.log({"pcc": error_pcc})

                            error_pcc_train = evaluate_pcc(final_model_sep, X_train, y_train)
                            print(f'pcc error train: {error_pcc_train}')
                            wandb.log({"train_pcc": error_pcc_train})

                            # Evaluate conditional metrics
                            above_threshold = mae_plus_threshold
                            error_mae_cond = evaluate_mae(
                                final_model_sep, X_test, y_test, above_threshold=above_threshold)
                            print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
                            wandb.log({"mae+": error_mae_cond})

                            error_mae_cond_train = evaluate_mae(
                                final_model_sep, X_train, y_train, above_threshold=above_threshold)
                            print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                            wandb.log({"train_mae+": error_mae_cond_train})

                            error_pcc_cond = evaluate_pcc(
                                final_model_sep, X_test, y_test, above_threshold=above_threshold)
                            print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                            wandb.log({"pcc+": error_pcc_cond})

                            error_pcc_cond_train = evaluate_pcc(
                                final_model_sep, X_train, y_train, above_threshold=above_threshold)
                            print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
                            wandb.log({"train_pcc+": error_pcc_cond_train})

                            # Process and plot SEP events
                            test_directory = root_dir + '/testing'
                            filenames = process_sep_events(
                                test_directory,
                                final_model_sep,
                                title=title,
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                show_avsp=True,
                                using_cme=True,
                                cme_speed_threshold=cme_speed_threshold)

                            for filename in filenames:
                                log_title = os.path.basename(filename)
                                wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                            test_directory = root_dir + '/training'
                            filenames = process_sep_events(
                                test_directory,
                                final_model_sep,
                                title=title,
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                show_avsp=True,
                                prefix='training',
                                using_cme=True,
                                cme_speed_threshold=cme_speed_threshold)

                            for filename in filenames:
                                log_title = os.path.basename(filename)
                                wandb.log({f'training_{log_title}': wandb.Image(filename)})

                            # Plot correlation distributions
                            file_path = plot_repr_corr_dist(
                                final_model_sep,
                                X_train_filtered, y_train_filtered,
                                title + "_training",
                                model_type='features_reg'
                            )
                            wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})

                            file_path = plot_repr_corr_dist(
                                final_model_sep,
                                X_test_filtered, y_test_filtered,
                                title + "_test",
                                model_type='features_reg'
                            )
                            wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})

                            # Plot t-SNE
                            stage1_file_path = plot_tsne_delta(
                                final_model_sep,
                                X_train_filtered, y_train_filtered, title,
                                'stage2_training',
                                model_type='features_reg',
                                save_tag=current_time, seed=seed)
                            wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})

                            stage1_file_path = plot_tsne_delta(
                                final_model_sep,
                                X_test_filtered, y_test_filtered, title,
                                'stage2_testing',
                                model_type='features_reg',
                                save_tag=current_time, seed=seed)
                            wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})

                            # Plot error histograms
                            filename = plot_error_hist(
                                final_model_sep,
                                X_train, y_train,
                                sample_weights=None,
                                title=title,
                                prefix='training')
                            wandb.log({"training_error_hist": wandb.Image(filename)})

                            filename = plot_error_hist(
                                final_model_sep,
                                X_test, y_test,
                                sample_weights=None,
                                title=title,
                                prefix='testing')
                            wandb.log({"testing_error_hist": wandb.Image(filename)})

                            # Finish wandb run
                            wandb.finish()


if __name__ == '__main__':
    main()
