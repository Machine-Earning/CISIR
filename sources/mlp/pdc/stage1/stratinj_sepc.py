# import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import (
    plot_tsne_delta,
    plot_repr_correlation,
    plot_repr_corr_dist,
    plot_repr_corr_density,
    evaluate_pcc_repr
)

from modules.reweighting.ImportanceWeighting import MDI
from modules.shared.sep_globals import *
from modules.training.cme_modeling import ModelBuilder
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_sep_ds,
    filter_ds_up,
    set_seed,
    load_folds_sep_ds,
    stratified_batch_dataset,
    create_mlp,
    initialize_results_dict,
)


def main():
    """
    Main function to run the PDC model
    :return:
    """
    # list the devices available
    devices = tf.config.list_physical_devices('GPU')
    print(f'devices: {devices}')
    # Define the dataset options, including the sharding policy
    mb = ModelBuilder()  # Model builder
    pm = TrainingPhaseManager()  # Training phase manager
    
    alphas = [(2.4, 2.4)] # a list in case I want to run multiple alphas
    alpha_pdc = alphas[0][0]
    n_trials = len(TRIAL_SEEDS)
    results = initialize_results_dict(n_trials)
    results['name'] = f'sepc_apdc{alpha_pdc:.2f}_mdi_s1_quad_nae'
    rho = RHO_PRE[0]

    for seed_idx, seed in enumerate(TRIAL_SEEDS):
        for alpha, alphaV in alphas:
            # PARAMS
            batch_size = BATCH_SIZE_PRE  # full dataset used
            print(f'batch size : {batch_size}')
            # Construct the title
            title = f'sepc_apdc{alpha_pdc:.2f}_mdi_s1_quad_nae_seed{seed}'

            # Replace any other characters that are not suitable for filenames (if any)
            title = title.replace(' ', '_').replace(':', '_')

            # Create a unique experiment name with a timestamp
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f'{title}_{current_time}'
            # Set the early stopping patience and learning rate as variables
            set_seed(seed, use_deterministic=False)
            epochs = EPOCHS
            patience = PATIENCE_PRE
            learning_rate = START_LR_PRE
            weight_decay = WEIGHT_DECAY_PRE
            normalized_weights = NORMALIZED_WEIGHTS
            lr_cb_patience = LR_CB_PATIENCE_PRE
            lr_cb_factor = LR_CB_FACTOR_PRE
            lr_cb_min_lr = LR_CB_MIN_LR_PRE
            lr_cb_min_delta = LR_CB_MIN_DELTA
            cvrg_metric = CVRG_METRIC
            cvrg_min_delta = CVRG_MIN_DELTA

            hiddens = MLP_HIDDENS
            hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
            pretraining = True
            embed_dim = EMBED_DIM
            dropout = DROPOUT_PRE
            activation = ACTIVATION
            norm = NORM
            skip_repr = SKIP_REPR

            reduce_lr_on_plateau = ReduceLROnPlateau(
                monitor=LR_CB_MONITOR,
                factor=lr_cb_factor,
                patience=lr_cb_patience,
                verbose=VERBOSE,
                min_delta=lr_cb_min_delta,
                min_lr=lr_cb_min_lr)

            bandwidth = BANDWIDTH
            skipped_layers = SKIPPED_LAYERS
            n_filter = N_FILTER  # number of samples to keep outside the threshold
            sep_threshold = SEP_THRESHOLD  # lower threshold for the delta_p
            mae_plus_threshold = SEP_THRESHOLD
            smoothing_method = SMOOTHING_METHOD
            window_size = WINDOW_SIZE_PRE  # allows margin of error of 10 epochs
            val_window_size = VAL_WINDOW_SIZE_PRE  # allows margin of error of 10 epochs

            # Initialize wandb
            wandb.init(project="2025-Papers-Repr-SEPC", name=experiment_name, config={
                "patience": patience,
                "learning_rate": learning_rate,
                "min_lr": LR_CB_MIN_LR_PRE,
                "factor_lr": LR_CB_FACTOR_PRE,
                "patience_lr": LR_CB_PATIENCE_PRE,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "hiddens": hiddens_str,
                "pretraining": "pdc",
                "seed": seed,
                "stage": 1,
                "dropout": dropout,
                "activation": "LeakyReLU",
                "norm": norm,
                "optimizer": "adam",
                "architecture": "mlp",
                "alpha": alpha,
                "alphaV": alphaV,
                "bandwidth": bandwidth,
                "skipped_layers": skipped_layers,
                "embed_dim": embed_dim,
                "ds_version": DS_VERSION,
                "n_filter": n_filter,
                "sep_threshold": sep_threshold,
                'smoothing_method': smoothing_method,
                'window_size': window_size, 
                'val_window_size': val_window_size,
                'normalized_weights': normalized_weights,
            })

            # set the root directory
            root_dir = DS_PATH
            # build the dataset
            X_train, y_train = build_sep_ds(
                root_dir + '/sep_10mev_training.csv',
                shuffle_data=True,
                random_state=seed
            )
            print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
            # build the test set
            X_test, y_test = build_sep_ds(
                root_dir + '/sep_10mev_testing.csv',
                shuffle_data=False,
                random_state=seed
            )
            print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

            # get the number of features
            n_features = X_train.shape[1]
            print(f'n_features: {n_features}')

            # Compute the sample weights
            delta_train = y_train
            print(f'delta_train.shape: {delta_train.shape}')
            print(f'rebalancing the training set...')
            train_weights_dict = MDI(
                X_train, delta_train, alpha=alpha, 
                bandwidth=bandwidth).label_importance_map
            print(f'done rebalancing the training set...')

            # filtering training and test sets for additional results
            X_train_filtered, y_train_filtered = filter_ds_up(
                X_train, y_train,
                high_threshold=sep_threshold,
                N=n_filter, seed=seed)
            X_test_filtered, y_test_filtered = filter_ds_up(
                X_test, y_test,
                high_threshold=sep_threshold,
                N=n_filter, seed=seed)

            # 4-fold cross-validation
            folds_optimal_epochs = []
            for fold_idx, (X_subtrain, y_subtrain, X_val, y_val) in enumerate(
                load_folds_sep_ds(
                    root_dir,
                    random_state=seed,
                    shuffle=True
                )
            ):
                print(f'Fold: {fold_idx}')
                # print all cme_files shapes
                print(
                    f'X_subtrain.shape: {X_subtrain.shape}, y_subtrain.shape: {y_subtrain.shape}')
                print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

                # Compute the sample weights for subtraining
                delta_subtrain = y_subtrain
                print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                print(f'rebalancing the subtraining set...')
                subtrain_weights_dict = MDI(
                    X_subtrain, delta_subtrain, alpha=alpha, 
                    bandwidth=bandwidth).label_importance_map
                print(f'subtraining set rebalanced.')

                # Compute the sample weights for validation
                delta_val = y_val
                print(f'delta_val.shape: {delta_val.shape}')
                print(f'rebalancing the validation set...')
                val_weights_dict = MDI(
                    X_val, delta_val, alpha=alphaV, 
                    bandwidth=bandwidth).label_importance_map
                print(f'validation set rebalanced.')

                # create the model
                model_sep = create_mlp(
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
                    weight_decay=weight_decay
                )
                model_sep.summary()

                # Define the EarlyStopping callback
                early_stopping = SmoothEarlyStopping(
                    monitor=cvrg_metric,
                    min_delta=cvrg_min_delta,
                    patience=patience,
                    verbose=VERBOSE,
                    restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                    smoothing_method=smoothing_method,  # 'moving_average'
                    smoothing_parameters={'window_size': window_size})  # 10

                model_sep.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss=lambda y_true, z_pred: mb.pdc_loss_vec(
                        y_true, z_pred,
                        phase_manager=pm,
                        train_sample_weights=subtrain_weights_dict,
                        val_sample_weights=val_weights_dict,
                    )
                )

                subtrain_ds, subtrain_steps = stratified_batch_dataset(
                    X_subtrain, y_subtrain, batch_size)

                history = model_sep.fit(
                    subtrain_ds,
                    steps_per_epoch=subtrain_steps,
                    epochs=epochs,
                    validation_data=(X_val, y_val),
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
                optimal_epoch = find_optimal_epoch_by_smoothing(
                    history.history[ES_CB_MONITOR],
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')
                folds_optimal_epochs.append(optimal_epoch)
                # wandb log the fold's optimal
                print(f'fold_{fold_idx}_best_epoch: {folds_optimal_epochs[-1]}')
                wandb.log({f'fold_{fold_idx}_best_epoch': folds_optimal_epochs[-1]})

            # determine the optimal number of epochs from the folds
            optimal_epochs = int(np.mean(folds_optimal_epochs))
            print(f'optimal_epochs: {optimal_epochs}')
            wandb.log({'optimal_epochs': optimal_epochs})

            # create the model
            final_model_sep = create_mlp(
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
                weight_decay=weight_decay
            )

            final_model_sep.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss=lambda y_true, z_pred: mb.pdc_loss_vec(
                    y_true, z_pred,
                    phase_manager=pm,
                    train_sample_weights=train_weights_dict,
                )
            )

            train_ds, train_steps = stratified_batch_dataset(
                X_train, y_train, batch_size)

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
            final_model_sep.save_weights(f"fm_{str(experiment_name)}.h5")
            # print where the model weights are saved
            print(f"Model weights are saved in fm_{str(experiment_name)}.h5")

            # Evaluate the model correlation on the test set
            error_pcc = evaluate_pcc_repr(final_model_sep, X_test, y_test)
            print(f'pcc error delta test: {error_pcc}')
            wandb.log({"jpcc": error_pcc})

            # Evaluate the model correlation on the training set
            error_pcc_train = evaluate_pcc_repr(final_model_sep, X_train, y_train)
            print(f'pcc error delta train: {error_pcc_train}')
            wandb.log({"train_jpcc": error_pcc_train})

            above_threshold = mae_plus_threshold  # norm_upper_t
            # evaluate pcc+ on the test set
            error_pcc_cond = evaluate_pcc_repr(
                final_model_sep, X_test, y_test, i_above_threshold=above_threshold)
            print(f'pcc error delta i>= {above_threshold} test: {error_pcc_cond}')
            wandb.log({"jpcc+": error_pcc_cond})

            # evaluate pcc+ on the training set
            error_pcc_cond_train = evaluate_pcc_repr(
                final_model_sep, X_train, y_train, i_above_threshold=above_threshold)
            print(f'pcc error delta i>= {above_threshold} train: {error_pcc_cond_train}')
            wandb.log({"train_jpcc+": error_pcc_cond_train})

            # Evaluate the model correlation with colored
            file_path = plot_repr_corr_dist(
                final_model_sep,
                X_train_filtered, y_train_filtered,
                title + "_training"
            )
            wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_repr_corr_dist(
                final_model_sep,
                X_test_filtered, y_test_filtered,
                title + "_test"
            )
            wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            # Log t-SNE plot
            # Log the training t-SNE plot to wandb
            stage1_file_path = plot_tsne_delta(
                final_model_sep,
                X_train_filtered, y_train_filtered, title,
                'stage1_training',
                model_type='features',
                save_tag=current_time, seed=seed)
            wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
            print('stage1_file_path: ' + stage1_file_path)

            # Log the testing t-SNE plot to wandb
            stage1_file_path = plot_tsne_delta(
                final_model_sep,
                X_test_filtered, y_test_filtered, title,
                'stage1_testing',
                model_type='features',
                save_tag=current_time, seed=seed)
            wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
            print('stage1_file_path: ' + stage1_file_path)

            # Evaluate the model correlation
            file_path = plot_repr_correlation(
                final_model_sep,
                X_train_filtered, y_train_filtered,
                title + "_training"
            )
            wandb.log({'representation_correlation_plot_train': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_repr_correlation(
                final_model_sep,
                X_test_filtered, y_test_filtered,
                title + "_test"
            )
            wandb.log({'representation_correlation_plot_test': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            # Evaluate the model correlation density
            file_path = plot_repr_corr_density(
                final_model_sep,
                X_train_filtered, y_train_filtered,
                title + "_training"
            )
            wandb.log({'representation_correlation_density_plot_train': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_repr_corr_density(
                final_model_sep,
                X_test_filtered, y_test_filtered,
                title + "_test"
            )
            wandb.log({'representation_correlation_density_plot_test': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            # Finish the wandb run
            wandb.finish()


if __name__ == '__main__':
    main()
