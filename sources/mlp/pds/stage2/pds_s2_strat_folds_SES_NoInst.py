import os
from datetime import datetime

import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_delta, plot_repr_corr_dist
from modules.reweighting.exDenseReweightsD import exDenseReweightsD
from modules.shared.globals import *
from modules.training.cme_modeling import ModelBuilder
from modules.training.phase_manager import TrainingPhaseManager, IsTraining
from modules.training.smooth_early_stopping import SmoothEarlyStopping, find_optimal_epoch_by_smoothing
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    mse_pcc,
    filter_ds,
    plot_error_hist,
    set_seed, stratified_batch_dataset,
)
from modules.training.utils import get_weight_path

# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Define the lookup dictionary
weight_paths = {
    # No inj
    # (False,
    #  0): '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240627-012209MLP_e0_5_e1_8_p_slopeFalse_PDS_bs4096_alpha0.20_CME0_features_noinj.h5',
    # inj all
    # (False,
    #  0): '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240627-030006MLP_e0_5_e1_8_p_slopeFalse_PDSinj_bs4096_alpha0.20_CME0_features_all.h5',
    # # inj min 
    (False,
     0):
    # '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240731-084027MLP_strat_e0_5_e4_4_p6_1_p_slopeFalse_PDSinj_bs4096_alpha0.50_CME0_features_inj.h5'
    # '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240803-153026MLP_strat_e0_5_e4_4_p6_1_p_PDSinj_bs4096_alpha2.50_rho0.00_features_inj.h5'
        '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240803-214949MLP_strat_e0_5_e4_4_p6_1_p_PDSinj_bs4096_alpha5.00_rho0.00_features_inj.h5'
    ,

}


def main():
    """
    Main function to run the E-MLP model
    :return:
    """
    mb = ModelBuilder()
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for inputs_to_use in INPUTS_TO_USE:
            for add_slope in ADD_SLOPE:
                for cme_speed_threshold in CME_SPEED_THRESHOLD:
                    for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in [(0.5, 1, 0.1, 0)]:
                        for freeze in [False]:
                            for rho in [1e-3]:
                                # PARAMS
                                outputs_to_use = OUTPUTS_TO_USE
                                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                                lambda_ = 3.3  # LAMBDA
                                # Construct the title
                                title = f'MLP_pdsS2_amse{alpha_mse:.2f}_SES_NoInst'

                                # Replace any other characters that are not suitable for filenames (if any)
                                title = title.replace(' ', '_').replace(':', '_')

                                # Create a unique experiment name with a timestamp
                                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                                experiment_name = f'{title}_{current_time}'

                                set_seed(seed)
                                patience = PATIENCE  # higher patience
                                learning_rate = START_LR_FT  # lower due to finetuning

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
                                hiddens = MLP_HIDDENS
                                proj_hiddens = PROJ_HIDDENS
                                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                                bandwidth = BANDWIDTH
                                repr_dim = REPR_DIM
                                output_dim = len(outputs_to_use)
                                dropout = 0.4  # DROPOUT
                                activation = ACTIVATION
                                norm = NORM
                                pds = True
                                cme_speed_threshold = cme_speed_threshold
                                weight_path = get_weight_path(weight_paths, add_slope, cme_speed_threshold)
                                residual = RESIDUAL
                                skipped_layers = SKIPPED_LAYERS
                                N = N_FILTERED  # number of samples to keep outside the threshold
                                lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                                upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                                mae_plus_threshold = MAE_PLUS_THRESHOLD
                                smoothing_method = 'moving_average'
                                window_size = 25  # allows margin of error of 10 epochs

                                # Initialize wandb
                                wandb.init(project="Report-Test", name=experiment_name, config={
                                    "inputs_to_use": inputs_to_use,
                                    "add_slope": add_slope,
                                    "patience": patience,
                                    "learning_rate": learning_rate,
                                    "weight_decay": weight_decay,
                                    "momentum_beta1": momentum_beta1,
                                    "batch_size": batch_size,
                                    "epochs": epochs,
                                    # hidden in a more readable format  (wandb does not support lists)
                                    "hiddens": hiddens_str,
                                    "lambda": lambda_,
                                    "seed": seed,
                                    "alpha_mse": alpha_mse,
                                    "alpha_pcc": alpha_pcc,
                                    "alphaV_mse": alphaV_mse,
                                    "alphaV_pcc": alphaV_pcc,
                                    "bandwidth": bandwidth,
                                    "repr_dim": repr_dim,
                                    "dropout": dropout,
                                    "activation": 'LeakyReLU',
                                    "norm": norm,
                                    'optimizer': 'adamw',
                                    'output_dim': output_dim,
                                    'architecture': 'mlp',
                                    "freeze": freeze,
                                    "pds": pds,
                                    "stage": 2,
                                    "stage1_weights": weight_path,
                                    "cme_speed_threshold": cme_speed_threshold,
                                    "residual": residual,
                                    "skipped_layers": skipped_layers,
                                    'ds_version': DS_VERSION,
                                    'mae_plus_th': mae_plus_threshold,
                                    'sam_rho': rho,
                                    'smoothing_method': smoothing_method,
                                    'window_size': window_size
                                })

                                # set the root directory
                                root_dir = DS_PATH
                                # build the dataset
                                X_train, y_train = build_dataset(
                                    root_dir + '/training',
                                    inputs_to_use=inputs_to_use,
                                    add_slope=add_slope,
                                    outputs_to_use=outputs_to_use,
                                    cme_speed_threshold=cme_speed_threshold,
                                    shuffle_data=True)
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

                                X_test, y_test = build_dataset(
                                    root_dir + '/testing',
                                    inputs_to_use=inputs_to_use,
                                    add_slope=add_slope,
                                    outputs_to_use=outputs_to_use,
                                    cme_speed_threshold=cme_speed_threshold)
                                # print the test set shapes
                                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                                # filtering training and test sets for additional results
                                X_train_filtered, y_train_filtered = filter_ds(
                                    X_train, y_train,
                                    low_threshold=lower_threshold,
                                    high_threshold=upper_threshold,
                                    N=N, seed=seed)
                                X_test_filtered, y_test_filtered = filter_ds(
                                    X_test, y_test,
                                    low_threshold=lower_threshold,
                                    high_threshold=upper_threshold,
                                    N=N, seed=seed)

                                # 4-fold cross-validation
                                # folds_optimal_epochs = []
                                # for fold_idx, (X_subtrain, y_subtrain, X_val, y_val) in enumerate(
                                #         stratified_4fold_split(X_train, y_train, seed=seed, shuffle=True)):
                                #     print(f'Fold: {fold_idx}')

                                # build subtraining and validation from paths
                                X_subtrain, y_subtrain = build_dataset(
                                    root_dir + '/subtraining',
                                    inputs_to_use=inputs_to_use,
                                    add_slope=add_slope,
                                    outputs_to_use=outputs_to_use,
                                    cme_speed_threshold=cme_speed_threshold,
                                    shuffle_data=True)

                                X_val, y_val = build_dataset(
                                    root_dir + '/validation',
                                    inputs_to_use=inputs_to_use,
                                    add_slope=add_slope,
                                    outputs_to_use=outputs_to_use,
                                    cme_speed_threshold=cme_speed_threshold)

                                # print all cme_files shapes
                                print(f'X_subtrain.shape: {X_subtrain.shape}, y_subtrain.shape: {y_subtrain.shape}')
                                print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}')

                                # Compute the sample weights for subtraining
                                delta_subtrain = y_subtrain[:, 0]
                                print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                                print(f'rebalancing the subtraining set...')
                                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_subtrain)
                                mse_subtrain_weights_dict = exDenseReweightsD(
                                    X_subtrain, delta_subtrain,
                                    alpha=alpha_mse, bw=bandwidth,
                                    min_norm_weight=min_norm_weight,
                                    debug=False).label_reweight_dict
                                pcc_subtrain_weights_dict = exDenseReweightsD(
                                    X_subtrain, delta_subtrain,
                                    alpha=alpha_pcc, bw=bandwidth,
                                    min_norm_weight=min_norm_weight,
                                    debug=False).label_reweight_dict
                                print(f'subtraining set rebalanced.')

                                # Compute the sample weights for validation
                                delta_val = y_val[:, 0]
                                print(f'delta_val.shape: {delta_val.shape}')
                                print(f'rebalancing the validation set...')
                                min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_val)
                                mse_val_weights_dict = exDenseReweightsD(
                                    X_val, delta_val,
                                    alpha=alphaV_mse, bw=bandwidth,
                                    min_norm_weight=min_norm_weight,
                                    debug=False).label_reweight_dict
                                pcc_val_weights_dict = exDenseReweightsD(
                                    X_val, delta_val,
                                    alpha=alphaV_pcc, bw=bandwidth,
                                    min_norm_weight=min_norm_weight,
                                    debug=False).label_reweight_dict
                                print(f'validation set rebalanced.')

                                # create the model
                                model_sep_stage1 = create_mlp(
                                    input_dim=n_features,
                                    hiddens=hiddens,
                                    output_dim=0,
                                    pds=pds,
                                    repr_dim=repr_dim,
                                    dropout_rate=dropout,
                                    activation=activation,
                                    norm=norm,
                                    residual=residual,
                                    skipped_layers=skipped_layers
                                )
                                model_sep_stage1.summary()

                                # load the weights from the first stage
                                print(f'weights loading from: {weight_path}')
                                model_sep_stage1.load_weights(weight_path)
                                # print the save
                                print(f'weights loaded successfully from: {weight_path}')

                                # Log t-SNE plot for training
                                # Log the training t-SNE plot to wandb
                                stage1_file_path = plot_tsne_delta(
                                    model_sep_stage1,
                                    X_train_filtered, y_train_filtered, title,
                                    'stage1_training',
                                    model_type='features',
                                    save_tag=current_time, seed=seed)
                                wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                                print('stage1_file_path: ' + stage1_file_path)

                                # Log t-SNE plot for testing
                                # Log the testing t-SNE plot to wandb
                                stage1_file_path = plot_tsne_delta(
                                    model_sep_stage1,
                                    X_test_filtered, y_test_filtered, title,
                                    'stage1_testing',
                                    model_type='features',
                                    save_tag=current_time, seed=seed)
                                wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                                print('stage1_file_path: ' + stage1_file_path)

                                model_sep = mb.add_proj_head(
                                    model_sep_stage1,
                                    output_dim=output_dim,
                                    freeze_features=freeze,
                                    pds=pds,
                                    hiddens=proj_hiddens,
                                    dropout_rate=dropout,
                                    activation=activation,
                                    norm=norm,
                                    residual=residual,
                                    skipped_layers=skipped_layers,
                                    name='mlp',
                                    sam_rho=rho
                                )
                                model_sep.summary()

                                # Define the EarlyStopping callback
                                # early_stopping = EarlyStopping(
                                #     monitor=ES_CB_MONITOR,
                                #     patience=patience,
                                #     verbose=VERBOSE,
                                #     restore_best_weights=ES_CB_RESTORE_WEIGHTS)

                                early_stopping = SmoothEarlyStopping(
                                    monitor=ES_CB_MONITOR,
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
                                        'forecast_head': lambda y_true, y_pred: mse_pcc(
                                            y_true, y_pred,
                                            phase_manager=pm,
                                            lambda_factor=lambda_,
                                            train_mse_weight_dict=mse_subtrain_weights_dict,
                                            train_pcc_weight_dict=pcc_subtrain_weights_dict,
                                            val_mse_weight_dict=mse_val_weights_dict,
                                            val_pcc_weight_dict=pcc_val_weights_dict,
                                        )
                                    }
                                )

                                # Step 1: Create stratified dataset for the subtraining and validation set
                                subtrain_ds, subtrain_steps = stratified_batch_dataset(
                                    X_subtrain, y_subtrain, batch_size)
                                val_ds, val_steps = stratified_batch_dataset(
                                    X_val, y_val, batch_size)

                                # Map the subtraining dataset to return {'output': y} format
                                subtrain_ds = subtrain_ds.map(lambda x, y: (x, {'forecast_head': y}))
                                val_ds = val_ds.map(lambda x, y: (x, {'forecast_head': y}))

                                # Train the model with the callback
                                history = model_sep.fit(
                                    subtrain_ds,
                                    steps_per_epoch=subtrain_steps,
                                    epochs=epochs, batch_size=batch_size,
                                    validation_data=val_ds,
                                    validation_steps=val_steps,
                                    callbacks=[
                                        early_stopping,
                                        reduce_lr_on_plateau,
                                        WandbCallback(save_model=WANDB_SAVE_MODEL),
                                        IsTraining(pm)
                                    ],
                                    verbose=VERBOSE
                                )

                                # optimal epoch for fold
                                # folds_optimal_epochs.append(np.argmin(history.history[ES_CB_MONITOR]) + 1)
                                # Use the quadratic fit function to find the optimal epoch
                                optimal_epoch = find_optimal_epoch_by_smoothing(
                                    history.history[ES_CB_MONITOR],
                                    smoothing_method=smoothing_method,
                                    smoothing_parameters={'window_size': window_size},
                                    mode='min')

                                # folds_optimal_epochs.append(optimal_epoch)
                                # wandb log the fold's optimal
                                # print(f'fold_{fold_idx}_best_epoch: {folds_optimal_epochs[-1]}')
                                # wandb.log({f'fold_{fold_idx}_best_epoch': folds_optimal_epochs[-1]})
                                print(f'optimal_epoch: {optimal_epoch}')
                                wandb.log({'optimal_epoch': optimal_epoch})

                                # determine the optimal number of epochs from the folds
                                # optimal_epochs = int(np.mean(folds_optimal_epochs))
                                optimal_epochs = int(optimal_epoch)
                                print(f'optimal_epochs: {optimal_epochs}')
                                wandb.log({'optimal_epochs': optimal_epochs})

                                final_model_sep_stage1 = create_mlp(
                                    input_dim=n_features,
                                    hiddens=hiddens,
                                    output_dim=0,
                                    pds=pds,
                                    repr_dim=repr_dim,
                                    dropout_rate=dropout,
                                    activation=activation,
                                    norm=norm,
                                    residual=residual,
                                    skipped_layers=skipped_layers
                                )
                                final_model_sep_stage1.load_weights(weight_path)

                                # Recreate the model architecture for final_model_sep
                                final_model_sep = mb.add_proj_head(
                                    final_model_sep_stage1,
                                    output_dim=output_dim,
                                    freeze_features=freeze,
                                    pds=pds,
                                    hiddens=proj_hiddens,
                                    dropout_rate=dropout,
                                    activation=activation,
                                    norm=norm,
                                    residual=residual,
                                    skipped_layers=skipped_layers,
                                    name='mlp',
                                    sam_rho=rho
                                )

                                final_model_sep.compile(
                                    optimizer=AdamW(
                                        learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        beta_1=momentum_beta1
                                    ),
                                    loss={
                                        'forecast_head': lambda y_true, y_pred: mse_pcc(
                                            y_true, y_pred,
                                            phase_manager=pm,
                                            lambda_factor=lambda_,
                                            train_mse_weight_dict=mse_train_weights_dict,
                                            train_pcc_weight_dict=pcc_train_weights_dict,
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
                                final_model_sep.save_weights(f"final_model_weights_{experiment_name}_s2min_reg.h5")
                                # print where the model weights are saved
                                print(f"Model weights are saved in final_model_weights_{experiment_name}_s2min_reg.h5")

                                # TODO: put the battery of evaluation in a function since they always repeat
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

                                # evaluate the model on test cme_files
                                above_threshold = mae_plus_threshold
                                # evaluate the model error for rare samples on test set
                                error_mae_cond = evaluate_mae(
                                    final_model_sep, X_test, y_test, above_threshold=above_threshold)
                                print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
                                wandb.log({"mae+": error_mae_cond})

                                # evaluate the model error for rare samples on training set
                                error_mae_cond_train = evaluate_mae(
                                    final_model_sep, X_train, y_train, above_threshold=above_threshold)
                                print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                                wandb.log({"train_mae+": error_mae_cond_train})

                                # evaluate the model correlation for rare samples on test set
                                error_pcc_cond = evaluate_pcc(
                                    final_model_sep, X_test, y_test, above_threshold=above_threshold)
                                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                                wandb.log({"pcc+": error_pcc_cond})

                                # evaluate the model correlation for rare samples on training set
                                error_pcc_cond_train = evaluate_pcc(
                                    final_model_sep, X_train, y_train, above_threshold=above_threshold)
                                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
                                wandb.log({"train_pcc+": error_pcc_cond_train})

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

                                # Evaluate the model correlation with colored
                                file_path = plot_repr_corr_dist(
                                    final_model_sep,
                                    X_train_filtered, y_train_filtered,
                                    title + "_training",
                                    model_type='features_reg'
                                )
                                wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                                print('file_path: ' + file_path)

                                file_path = plot_repr_corr_dist(
                                    final_model_sep,
                                    X_test_filtered, y_test_filtered,
                                    title + "_test",
                                    model_type='features_reg'
                                )
                                wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                                print('file_path: ' + file_path)

                                # Log t-SNE plot
                                # Log the training t-SNE plot to wandb
                                stage1_file_path = plot_tsne_delta(
                                    final_model_sep,
                                    X_train_filtered, y_train_filtered, title,
                                    'stage2_training',
                                    model_type='features_reg',
                                    save_tag=current_time, seed=seed)
                                wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                                print('stage1_file_path: ' + stage1_file_path)

                                # Log the testing t-SNE plot to wandb
                                stage1_file_path = plot_tsne_delta(
                                    final_model_sep,
                                    X_test_filtered, y_test_filtered, title,
                                    'stage2_testing',
                                    model_type='features_reg',
                                    save_tag=current_time, seed=seed)
                                wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                                print('stage1_file_path: ' + stage1_file_path)

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

                                # Finish the wandb run
                                wandb.finish()


if __name__ == '__main__':
    main()
