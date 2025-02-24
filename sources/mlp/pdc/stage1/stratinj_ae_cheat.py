# import os
from datetime import datetime

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import (
    plot_tsne_delta,
    plot_repr_correlation,
    plot_repr_corr_dist,
    plot_repr_corr_density,
    evaluate_pcc_repr
)
from modules.reweighting.ImportanceWeighting import ReciprocalImportance
from modules.shared.globals import *
from modules.training.cme_modeling import ModelBuilder
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    cmse,
    build_dataset,
    filter_ds,
    set_seed,
    stratified_batch_dataset,
    create_mlp,
    add_decoder
)


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # left is 1


def main():
    """
    Main function to run the PDS model
    :return:
    """
    # list the devices available
    devices = tf.config.list_physical_devices('GPU')
    print(f'devices: {devices}')
    # Define the dataset options, including the sharding policy
    mb = ModelBuilder()  # Model builder
    pm = TrainingPhaseManager()  # Training phase manager

    for seed in SEEDS:
        for alpha, alphaV in REWEIGHTS_PRE:
            for rho in RHO_PRE:
                # PARAMS
                inputs_to_use = INPUTS_TO_USE[0]
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]
                add_slope = ADD_SLOPE[0]
                outputs_to_use = OUTPUTS_TO_USE
                batch_size = BATCH_SIZE_PRE  # full dataset used
                lambda_ = 5e-5  # for cmse with reciprocal importances
                print(f'batch size : {batch_size}')

                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                # Construct the title
                title = f'mlp2ae_pdcStratInj_bs{batch_size}_cheat'

                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')

                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                epochs = EPOCHS
                patience = PATIENCE_PRE
                learning_rate = START_LR_PRE
                weight_decay = WEIGHT_DECAY_PRE
                momentum_beta1 = MOMENTUM_BETA1

                hiddens = MLP_HIDDENS
                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                pretraining = True
                embed_dim = EMBED_DIM
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                skip_repr = SKIP_REPR
                lr_cb_patience = LR_CB_PATIENCE
                lr_cb_factor = LR_CB_FACTOR
                lr_cb_min_lr = LR_CB_MIN_LR_PRE
                lr_cb_min_delta = LR_CB_MIN_DELTA
                cvrg_metric = CVRG_METRIC
                cvrg_min_delta = CVRG_MIN_DELTA

                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor=LR_CB_MONITOR,
                    factor=lr_cb_factor,
                    patience=lr_cb_patience,
                    verbose=VERBOSE,
                    min_delta=lr_cb_min_delta,
                    min_lr=lr_cb_min_lr)

                bandwidth = BANDWIDTH
                residual = RESIDUAL
                skipped_layers = SKIPPED_LAYERS
                N = N_FILTERED  # number of samples to keep outside the threshold
                lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                mae_plus_threshold = MAE_PLUS_THRESHOLD
                smoothing_method = SMOOTHING_METHOD
                window_size = WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = VAL_WINDOW_SIZE  # allows margin of error of 10 epochs
                # Set the reconstruction loss weight
                reconstruction_loss_weight = AE_LAMBDA  # Adjust as needed

                # Initialize wandb
                wandb.init(project='Repr-Jan-Report', name=experiment_name, config={
                    'inputs_to_use': inputs_to_use,
                    'add_slope': add_slope,
                    'patience': patience,
                    'learning_rate': learning_rate,
                    'min_lr': LR_CB_MIN_LR_PRE,
                    'weight_decay': weight_decay,
                    'momentum_beta1': momentum_beta1,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    # hidden in a more readable format  (wandb does not support lists)
                    'hiddens': hiddens_str,
                    'pretraining': pretraining,
                    'seed': seed,
                    'stage': 1,
                    'dropout': dropout,
                    'activation': 'LeakyReLU',
                    'norm': norm,
                    'optimizer': 'adamw',
                    'architecture': 'mlp',
                    'alpha': alpha,
                    'alphaVal': alphaV,
                    'bandwidth': bandwidth,
                    'residual': residual,
                    'skipped_layers': skipped_layers,
                    'embed_dim': embed_dim,
                    'ds_version': DS_VERSION,
                    'N_freq': N,
                    'lower_t': lower_threshold,
                    'upper_t': upper_threshold,
                    'mae_plus_th': mae_plus_threshold,
                    'cme_speed_threshold': cme_speed_threshold,
                    'sam_rho': rho,
                    'outputs_to_use': outputs_to_use,
                    'inj': 'strat',
                    'smoothing_method': smoothing_method,
                    'window_size': window_size,
                    'val_window_size': val_window_size,
                    'reconstruction_loss_weight': reconstruction_loss_weight,
                    'lr_cb_patience': lr_cb_patience,
                    'lr_cb_factor': lr_cb_factor,
                    'lr_cb_min_lr': lr_cb_min_lr,
                    'lr_cb_min_delta': lr_cb_min_delta,
                    'cvrg_metric': cvrg_metric,
                    'cvrg_min_delta': cvrg_min_delta
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

                # print all cme_files shapes
                print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

                # get the number of features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                # pds normalize the data
                # y_train_norm, norm_lower_t, norm_upper_t = pds_space_norm(y_train)

                # Compute the sample weights
                delta_train = y_train[:, 0]
                print(f'delta_train.shape: {delta_train.shape}')
                print(f'rebalancing the training set...')
                train_weights_dict = ReciprocalImportance(
                    X_train, delta_train,
                    alpha=alpha,
                    bandwidth=bandwidth).label_importance_map
                print(f'done rebalancing the training set...')

                X_train_filtered, y_train_filtered = filter_ds(
                    X_train, y_train,
                    low_threshold=lower_threshold,
                    high_threshold=upper_threshold,
                    N=N, seed=seed)

                # build the test set
                X_test, y_test, _, _ = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                # y_test_norm, _, _ = pds_space_norm(y_test)

                # get the reweights for test set
                delta_test = y_test[:, 0]
                print(f'delta_test.shape: {delta_test.shape}')
                print(f'rebalancing the test set...')
                test_weights_dict = ReciprocalImportance(
                    X_test, delta_test,
                    alpha=alphaV,
                    bandwidth=bandwidth).label_importance_map

                X_test_filtered, y_test_filtered = filter_ds(
                    X_test, y_test,
                    low_threshold=lower_threshold,
                    high_threshold=upper_threshold,
                    N=N, seed=seed)

                # create the model
                encoder_model = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    output_dim=0,
                    pretraining=pretraining,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho,
                )

                # Add decoder to create the autoencoder model
                model_sep = add_decoder(
                    encoder_model=encoder_model,
                    hiddens=hiddens,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    skip_connections=(skipped_layers > 0)
                )

                model_sep.summary()

                # Define the EarlyStopping callback
                early_stopping = SmoothEarlyStopping(
                    monitor=CVRG_METRIC,
                    min_delta=CVRG_MIN_DELTA,
                    patience=patience,
                    verbose=VERBOSE,
                    restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                    smoothing_method=smoothing_method,  # 'moving_average'
                    smoothing_parameters={'window_size': window_size})  # 10

                model_sep.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss=[
                        lambda y_true, y_pred: mb.pdc_loss_linear_vec(
                            y_true, y_pred,
                            phase_manager=pm,
                            train_sample_weights=train_weights_dict,
                            val_sample_weights=test_weights_dict,
                        ),
                        lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_,
                            train_mse_weight_dict=train_weights_dict,
                            val_mse_weight_dict=test_weights_dict,
                        )
                    ],
                    loss_weights=[
                        1.0,
                        reconstruction_loss_weight
                    ]
                )

                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Adjust the datasets to include both y and X for reconstruction
                train_ds = train_ds.map(lambda x, y: (x, (y, x)))
                val_data = (X_test, (y_test, X_test))

                history = model_sep.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=epochs,
                    validation_data=val_data,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        early_stopping,
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # optimal epoch for fold
                # folds_optimal_epochs.append(np.argmin(history.history[ES_CB_MONITOR]) + 1)
                # Use the quadratic fit function to find the optimal epoch
                optimal_epochs = find_optimal_epoch_by_smoothing(
                    history.history[ES_CB_MONITOR],
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')

                print(f'optimal_epochs: {optimal_epochs}')
                wandb.log({'optimal_epochs': optimal_epochs})

                # create the model
                final_encoder = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    output_dim=0,
                    pretraining=pretraining,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho,
                )

                # Add decoder to create the autoencoder model
                final_model_sep = add_decoder(
                    encoder_model=final_encoder,
                    hiddens=hiddens,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    skip_connections=(skipped_layers > 0)
                )

                final_model_sep.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss=[
                        lambda y_true, y_pred: mb.pdc_loss_linear_vec(
                            y_true, y_pred,
                            phase_manager=pm,
                            train_sample_weights=train_weights_dict,
                        ),
                        lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_,
                            train_mse_weight_dict=train_weights_dict,
                        )
                    ],
                    loss_weights=[
                        1.0,
                        reconstruction_loss_weight
                    ]
                )

                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Adjust the dataset to include both y_train and X_train for reconstruction
                train_ds = train_ds.map(lambda x, y: (x, (y, x)))

                final_model_sep.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=optimal_epochs,
                    batch_size=batch_size,
                    callbacks=[
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Save the final model
                final_encoder.save_weights(f"final_model_weights_{str(experiment_name)}.h5")
                # print where the model weights are saved
                print(f"Model weights are saved in final_model_weights_{str(experiment_name)}.h5")

                above_threshold = mae_plus_threshold  # norm_upper_t
                # evaluate pcc+ on the test set
                error_pcc_cond = evaluate_pcc_repr(
                    final_encoder, X_test, y_test, i_above_threshold=above_threshold)
                print(f'pcc error delta i>= {above_threshold} test: {error_pcc_cond}')
                wandb.log({"jpcc+": error_pcc_cond})

                # evaluate pcc+ on the training set
                error_pcc_cond_train = evaluate_pcc_repr(
                    final_encoder, X_train, y_train, i_above_threshold=above_threshold)
                print(f'pcc error delta i>= {above_threshold} train: {error_pcc_cond_train}')
                wandb.log({"train_jpcc+": error_pcc_cond_train})

                # Evaluate the model correlation on the test set
                error_pcc = evaluate_pcc_repr(final_encoder, X_test, y_test)
                print(f'pcc error delta test: {error_pcc}')
                wandb.log({"jpcc": error_pcc})

                # Evaluate the model correlation on the training set
                error_pcc_train = evaluate_pcc_repr(final_encoder, X_train, y_train)
                print(f'pcc error delta train: {error_pcc_train}')
                wandb.log({"train_jpcc": error_pcc_train})

                # Evaluate the model correlation with colored
                file_path = plot_repr_corr_dist(
                    final_encoder,
                    X_train_filtered, y_train_filtered,
                    title + "_training"
                )
                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_corr_dist(
                    final_encoder,
                    X_test_filtered, y_test_filtered,
                    title + "_test"
                )
                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Log t-SNE plot
                # Log the training t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    final_encoder,
                    X_train_filtered, y_train_filtered, title,
                    'stage1_training',
                    model_type='features',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Log the testing t-SNE plot to wandb
                stage1_file_path = plot_tsne_delta(
                    final_encoder,
                    X_test_filtered, y_test_filtered, title,
                    'stage1_testing',
                    model_type='features',
                    save_tag=current_time, seed=seed)
                wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Evaluate the model correlation
                file_path = plot_repr_correlation(
                    final_encoder,
                    X_train_filtered, y_train_filtered,
                    title + "_training"
                )
                wandb.log({'representation_correlation_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_correlation(
                    final_encoder,
                    X_test_filtered, y_test_filtered,
                    title + "_test"
                )
                wandb.log({'representation_correlation_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Evaluate the model correlation density
                file_path = plot_repr_corr_density(
                    final_encoder,
                    X_train_filtered, y_train_filtered,
                    title + "_training"
                )
                wandb.log({'representation_correlation_density_plot_train': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                file_path = plot_repr_corr_density(
                    final_encoder,
                    X_test_filtered, y_test_filtered,
                    title + "_test"
                )
                wandb.log({'representation_correlation_density_plot_test': wandb.Image(file_path)})
                print('file_path: ' + file_path)

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
