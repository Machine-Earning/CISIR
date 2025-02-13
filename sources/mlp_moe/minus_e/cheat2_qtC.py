import os
from datetime import datetime

import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.reweighting.exQtCReweightsD import exDenseReweightsD
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
    get_minus_cls
)


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    """
    Main function to run the E-MLP model for data with labels <= lower threshold
    :return:
    """

    # set the training phase manager - necessary for mse + pcc loss
    pm = TrainingPhaseManager()

    inputs_to_use = INPUTS_TO_USE[0]  # Use first element
    cme_speed_threshold = CME_SPEED_THRESHOLD[0]  # Use first element
    add_slope = ADD_SLOPE[0]  # Use first element

    # Path to pre-trained model weights
    pretrained_weights = None #PRE_WEIGHT_PATH

    for seed in SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in [(1.1, 1.1, 1.1, 1.1)]:
            for rho in RHO_MOE_M:  # SAM_RHOS:
                # PARAMS
                outputs_to_use = OUTPUTS_TO_USE
                lambda_factor = LAMBDA_FACTOR_MOE_M  # lambda for the loss


                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                # inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'mlp2_amse{alpha_mse:.2f}_minus_e_qtC'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp

                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = PATIENCE_MOE_M  # higher patience
                learning_rate = START_LR_MOE_M  # starting learning rate
                asym_type = ASYM_TYPE_MOE
                lr_cb_min_lr = LR_CB_MIN_LR
                lr_cb_factor = LR_CB_FACTOR
                lr_cb_patience = LR_CB_PATIENCE
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

                weight_decay = WEIGHT_DECAY  # 1e-5 # higher weight decay
                momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                batch_size = BATCH_SIZE_MOE  # higher batch size
                epochs = EPOCHS  # higher epochs
                hiddens = MLP_HIDDENS  # hidden layers
                proj_hiddens = PROJ_HIDDENS  # projection hidden layers

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = len(outputs_to_use)
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                cme_speed_threshold = cme_speed_threshold
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
                pretraining = False

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
                    "importance_type": 'cosine',
                    # hidden in a more readable format  (wandb does not support lists)
                    "hiddens": hiddens_str,
                    "loss": 'mse_pcc',
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
                    'architecture': 'mlp_res_repr',
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
                    'lower_threshold': lower_threshold,
                    'upper_threshold': upper_threshold,
                    'cvrg_metric': cvrg_metric,
                    'cvrg_min_delta': cvrg_min_delta,
                    'pretrained_weights': pretrained_weights,
                    'lr_cb_min_lr': lr_cb_min_lr,
                    'lr_cb_factor': lr_cb_factor,
                    'lr_cb_patience': lr_cb_patience,
                    'lr_cb_min_delta': lr_cb_min_delta
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

                # Filter training data to only include samples where label <= lower_threshold
                X_train, y_train, logI_train, logI_prev_train = get_minus_cls(
                    X_train, y_train, lower_threshold, logI_train, logI_prev_train)

                # print the training set shapes
                print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')
                # getting the reweights for training set
                delta_train = y_train[:, 0]
                print(f'delta_train.shape: {delta_train.shape}')
                print(f'rebalancing the training set...')
                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                mse_train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_mse, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                pcc_train_weights_dict = exDenseReweightsD(
                    X_train, delta_train,
                    alpha=alpha_pcc, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                print(f'training set rebalanced.')
                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                X_test, y_test, logI_test, logI_prev_test = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)

                # Filter test data to only include samples where label <= lower_threshold
                X_test, y_test, logI_test, logI_prev_test = get_minus_cls(
                    X_test, y_test, lower_threshold, logI_test, logI_prev_test)

                # print the test set shapes
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                # Compute the sample weights for test set
                delta_test = y_test[:, 0]
                print(f'delta_test.shape: {delta_test.shape}')
                print(f'rebalancing the test set...')
                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                mse_test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_mse, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                pcc_test_weights_dict = exDenseReweightsD(
                    X_test, delta_test,
                    alpha=alphaV_pcc, bw=bandwidth,
                    min_norm_weight=min_norm_weight,
                    debug=False).label_reweight_dict
                print(f'test set rebalanced.')

                # create the model for validation
                model_sep = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    output_dim=output_dim,
                    pretraining=pretraining,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho
                )
                model_sep.summary()
                if pretrained_weights is not None:
                    model_sep.load_weights(pretrained_weights)

                # Define the EarlyStopping callback
                early_stopping = SmoothEarlyStopping(
                    monitor=cvrg_metric,
                    min_delta=cvrg_min_delta,
                    patience=patience,
                    verbose=VERBOSE,
                    restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                    smoothing_method=smoothing_method,  # 'moving_average'
                    smoothing_parameters={'window_size': window_size})  # 10

                # Compile the model with the specified learning rate
                model_sep.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={
                        'forecast_head': lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_factor,
                            train_mse_weight_dict=mse_train_weights_dict,
                            train_pcc_weight_dict=None,
                            val_mse_weight_dict=mse_test_weights_dict,
                            val_pcc_weight_dict=None,
                            asym_type=asym_type

                        )
                    }
                )

                # Step 1: Create stratified dataset for the training set only
                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Map the training dataset to return {'output': y} format
                train_ds = train_ds.map(lambda x, y: (x, {'forecast_head': y}))

                # Prepare test data without batching
                test_data = (X_test, {'forecast_head': y_test})

                # Train the model with the callback
                history = model_sep.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=test_data,
                    callbacks=[
                        early_stopping,
                        reduce_lr_on_plateau,
                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                        IsTraining(pm)
                    ],
                    verbose=VERBOSE
                )

                # Use the quadratic fit function to find the optimal epoch
                optimal_epochs = find_optimal_epoch_by_smoothing(
                    history.history[ES_CB_MONITOR],
                    smoothing_method=smoothing_method,
                    smoothing_parameters={'window_size': val_window_size},
                    mode='min')
                print(f'optimal_epochs: {optimal_epochs}')
                wandb.log({'optimal_epochs': optimal_epochs})

                # create the model for validation
                final_model_sep = create_mlp(
                    input_dim=n_features,
                    hiddens=hiddens,
                    output_dim=output_dim,
                    pretraining=pretraining,
                    embed_dim=embed_dim,
                    dropout=dropout,
                    activation=activation,
                    norm=norm,
                    skip_repr=skip_repr,
                    skipped_layers=skipped_layers,
                    sam_rho=rho
                )
                final_model_sep.summary()
                if pretrained_weights is not None:
                    final_model_sep.load_weights(pretrained_weights)

                # final_model_sep.summary()
                final_model_sep.compile(
                    optimizer=AdamW(
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        beta_1=momentum_beta1
                    ),
                    loss={
                        'forecast_head': lambda y_true, y_pred: cmse(
                            y_true, y_pred,
                            phase_manager=pm,
                            lambda_factor=lambda_factor,
                            train_mse_weight_dict=mse_train_weights_dict,
                            train_pcc_weight_dict=None,
                            asym_type=asym_type
                        )
                    },
                )  # Compile the model just like before

                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Map the training dataset to return {'output': y} format
                train_ds = train_ds.map(lambda x, y: (x, {'forecast_head': y}))

                # Train on the full dataset
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
                final_model_sep.save_weights(f"final_model_weights_{experiment_name}_reg.h5")
                # print where the model weights are saved
                print(f"Model weights are saved in final_model_weights_{experiment_name}_reg.h5")

                # evaluate the model error on test set
                error_mae = evaluate_mae(final_model_sep, X_test, y_test)
                print(f'mae error: {error_mae}')
                wandb.log({"mae": error_mae})

                # evaluate the model error on training set
                error_mae_train = evaluate_mae(final_model_sep, X_train, y_train)
                print(f'mae error train: {error_mae_train}')
                wandb.log({"train_mae": error_mae_train})

                # evaluate the model correlation on test set
                error_pcc = evaluate_pcc(final_model_sep, X_test, y_test)
                print(f'pcc error: {error_pcc}')
                wandb.log({"pcc": error_pcc})

                # evaluate the model correlation on training set
                error_pcc_train = evaluate_pcc(final_model_sep, X_train, y_train)
                print(f'pcc error train: {error_pcc_train}')
                wandb.log({"train_pcc": error_pcc_train})

                # evaluate the model correlation on test set based on logI and logI_prev
                error_pcc_logI = evaluate_pcc(final_model_sep, X_test, y_test, logI_test, logI_prev_test)
                print(f'pcc error logI: {error_pcc_logI}')
                wandb.log({"pcc_I": error_pcc_logI})

                # evaluate the model correlation on training set based on logI and logI_prev
                error_pcc_logI_train = evaluate_pcc(final_model_sep, X_train, y_train, logI_train, logI_prev_train)
                print(f'pcc error logI train: {error_pcc_logI_train}')
                wandb.log({"train_pcc_I": error_pcc_logI_train})


                # Process SEP event files in the specified directory
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

                # Log the plot to wandb
                for filename in filenames:
                    log_title = os.path.basename(filename)
                    wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                # Process SEP event files in the specified directory
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

                # Log the plot to wandb
                for filename in filenames:
                    log_title = os.path.basename(filename)
                    wandb.log({f'training_{log_title}': wandb.Image(filename)})


                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
