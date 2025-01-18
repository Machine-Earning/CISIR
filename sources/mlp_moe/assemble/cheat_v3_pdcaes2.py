import os
from datetime import datetime

import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

# from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
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
    # filter_ds,
    # create_mlp,
    create_mlp_moe
)


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    # set the training phase manager - necessary for mse + pcc loss'
    combiner_pretrained_weights = COMBINER_PDCAE_S2
    pm = TrainingPhaseManager()

    for seed in SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS_MOE:
            for rho in RHO_MOE:  # SAM
                inputs_to_use = INPUTS_TO_USE[0]
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]
                add_slope = ADD_SLOPE[0]
                # PARAMS
                outputs_to_use = OUTPUTS_TO_USE
                lambda_factor = LAMBDA_FACTOR_MOE  # lambda for the loss
                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                # Construct the title
                title = f'mlp2pdcaes2_amse{alpha_mse:.2f}_moe_cheat_v3_randInitCombiner'
                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')
                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                set_seed(seed)
                patience = PATIENCE_MOE  # higher patience
                learning_rate = START_LR_MOE  # starting learning rate
                asym_type = ASYM_TYPE_MOE

                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor=LR_CB_MONITOR,
                    factor=LR_CB_FACTOR_MOE,
                    patience=LR_CB_PATIENCE_MOE,
                    verbose=VERBOSE,
                    min_delta=LR_CB_MIN_DELTA,
                    min_lr=LR_CB_MIN_LR_MOE)

                weight_decay = WEIGHT_DECAY_MOE  # 1e-5 # higher weight decay
                momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                batch_size = BATCH_SIZE  # higher batch size
                epochs = EPOCHS  # higher epochs
                hiddens = MLP_HIDDENS  # hidden layers
                proj_hiddens = PROJ_HIDDENS

                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = len(outputs_to_use)
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                cme_speed_threshold = cme_speed_threshold
                skip_repr = SKIP_REPR
                skipped_layers = SKIPPED_LAYERS
                # NOTE: no need for filtering for moe since cannot run t-SNE and repr correlation
                # N = N_FILTERED  # number of samples to keep outside the threshold
                # lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                # upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                mae_plus_threshold = MAE_PLUS_THRESHOLD
                smoothing_method = SMOOTHING_METHOD
                window_size = WINDOW_SIZE  # allows margin of error of 10 epochs
                val_window_size = VAL_WINDOW_SIZE  # allows margin of error of 10 epochs
                pretraining = True
                freeze_experts = FREEZE_EXPERT

                expert_paths = {
                    'plus': POS_EXPERT_PATH,
                    'zero': NZ_EXPERT_PATH,
                    'minus': NEG_EXPERT_PATH,
                    # 'combiner': COMBINER_PATH
                }

                # Initialize wandb
                wandb.init(project="Jan-Report", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    'min_lr': LR_CB_MIN_LR_MOE,
                    "weight_decay": weight_decay,
                    "momentum_beta1": momentum_beta1,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    # hidden in a more readable format  (wandb does not support lists)
                    "hiddens": hiddens_str,
                    "loss": 'cmse',
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
                    'architecture': 'mlp_moe',
                    'cme_speed_threshold': cme_speed_threshold,
                    'skip_repr': skip_repr,
                    'ds_version': DS_VERSION,
                    'mae_plus_th': mae_plus_threshold,
                    'sam_rho': rho,
                    'smoothing_method': smoothing_method,
                    'window_size': window_size,
                    'val_window_size': val_window_size,
                    'expert+_path': POS_EXPERT_PATH,
                    'expert0_path': NZ_EXPERT_PATH,
                    'expert-_path': NEG_EXPERT_PATH,
                    'combiner_path': COMBINER_PATH,
                    'combiner_pretrained_weights': combiner_pretrained_weights,
                    'asym_type': asym_type,
                    'freeze_experts': freeze_experts
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
                # print the test set shapes
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')
                # getting the reweights for test set
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

                # NOTE: no need for filtering for moe since cannot run t-SNE and repr correlation
                # # filtering training and test sets for additional results
                # X_train_filtered, y_train_filtered = filter_ds(
                #     X_train, y_train,
                #     low_threshold=lower_threshold,
                #     high_threshold=upper_threshold,
                #     N=N, seed=seed)
                # X_test_filtered, y_test_filtered = filter_ds(
                #     X_test, y_test,
                #     low_threshold=lower_threshold,
                #     high_threshold=upper_threshold,
                #     N=N, seed=seed)

                combiner_pretrained_config = {
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
                    'proj_hiddens': proj_hiddens,
                    'proj_neck': True,
                    'no_head': False
                }

                # create the model
                init_model_sep = create_mlp_moe(
                    hiddens=hiddens,
                    combiner_hiddens=hiddens,
                    input_dim=n_features,
                    embed_dim=embed_dim,
                    skipped_layers=skipped_layers,
                    skip_repr=skip_repr,
                    pretraining=pretraining,
                    freeze_experts=freeze_experts,
                    expert_paths=expert_paths,
                    combiner_pretrained_weights=combiner_pretrained_weights,
                    combiner_pretrained_config=combiner_pretrained_config,
                    mode=MODE_MOE,
                    activation=activation,
                    norm=norm,
                    sam_rho=rho
                )
                init_model_sep.summary()

                # Define the EarlyStopping callback
                early_stopping = SmoothEarlyStopping(
                    monitor=CVRG_METRIC,
                    min_delta=CVRG_MIN_DELTA,
                    patience=patience,
                    verbose=VERBOSE,
                    restore_best_weights=ES_CB_RESTORE_WEIGHTS,
                    smoothing_method=smoothing_method,  # 'moving_average'
                    smoothing_parameters={'window_size': window_size})  # 10

                # Compile the model with the specified learning rate
                init_model_sep.compile(
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
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            val_mse_weight_dict=mse_test_weights_dict,
                            val_pcc_weight_dict=pcc_test_weights_dict,
                            asym_type=asym_type
                        )
                    }
                )

                # Step 1: Create stratified dataset for the subtraining and validation set
                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)
                test_ds, test_steps = stratified_batch_dataset(
                    X_test, y_test, batch_size)

                # Map the training dataset to return {'output': y} format
                train_ds = train_ds.map(lambda x, y: (x, {'forecast_head': y}))
                test_ds = test_ds.map(lambda x, y: (x, {'forecast_head': y}))

                # Train the model with the callback
                history = init_model_sep.fit(
                    train_ds,
                    steps_per_epoch=train_steps,
                    epochs=epochs, batch_size=batch_size,
                    validation_data=test_ds,
                    validation_steps=test_steps,
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

                # Recreate and recompile the model for optimal epoch training
                final_model_sep = create_mlp_moe(
                    hiddens=hiddens,
                    combiner_hiddens=hiddens,
                    input_dim=n_features,
                    embed_dim=embed_dim,
                    skipped_layers=skipped_layers,
                    skip_repr=skip_repr,
                    pretraining=pretraining,
                    freeze_experts=freeze_experts,
                    expert_paths=expert_paths,
                    combiner_pretrained_weights=combiner_pretrained_weights,
                    combiner_pretrained_config=combiner_pretrained_config,
                    mode=MODE_MOE,
                    activation=activation,
                    norm=norm,
                    sam_rho=rho
                )

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
                            train_pcc_weight_dict=pcc_train_weights_dict,
                            asym_type=asym_type
                        )
                    }
                )

                # Step 1: Create stratified dataset for the subtraining and validation set
                train_ds, train_steps = stratified_batch_dataset(
                    X_train, y_train, batch_size)

                # Map the training dataset to return {'output': y} format
                train_ds = train_ds.map(lambda x, y: (x, {'forecast_head': y}))

                # Train to the optimal epoch
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
                final_model_sep.save_weights(f"final_model_moe_weights_{experiment_name}_reg.h5")
                # print where the model weights are saved
                print(f"Model weights are saved in final_model_moe_weights_{experiment_name}_reg.h5")

                # Save the combiner sub-model weights
                combiner_submodel = final_model_sep.get_layer("combiner")
                combiner_submodel.save_weights(f"combiner_v3_weights_{experiment_name}.h5")
                print(f"Combiner sub-model weights saved to combiner_v3_weights_{experiment_name}.h5")

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

                # evaluate the model correlation for rare samples on test set based on logI and logI_prev
                error_pcc_cond_logI = evaluate_pcc(
                    final_model_sep, X_test, y_test, logI_test, logI_prev_test, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond_logI}')
                wandb.log({"pcc+_I": error_pcc_cond_logI})

                # evaluate the model correlation for rare samples on training set based on logI and logI_prev
                error_pcc_cond_logI_train = evaluate_pcc(
                    final_model_sep, X_train, y_train, logI_train, logI_prev_train, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_logI_train}')
                wandb.log({"train_pcc+_I": error_pcc_cond_logI_train})

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

                # NOTE: cannot run plot_repr_corr_dist for moe
                # # Evaluate the model correlation with colored
                # file_path = plot_repr_corr_dist(
                #     final_model_sep,
                #     X_train_filtered, y_train_filtered,
                #     title + "_training",
                #     model_type='features_reg'
                # )
                # wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                # print('file_path: ' + file_path)

                # file_path = plot_repr_corr_dist(
                #     final_model_sep,
                #     X_test_filtered, y_test_filtered,
                #     title + "_test",
                #     model_type='features_reg'
                # )
                # wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                # print('file_path: ' + file_path)

                # NOTE: cannot run t-SNE for moe
                # # Log t-SNE plot 
                # # Log the training t-SNE plot to wandb
                # stage1_file_path = plot_tsne_delta(
                #     final_model_sep,
                #     X_train_filtered, y_train_filtered, title,
                #     'stage2_training',
                #     model_type='features_reg',
                #     save_tag=current_time, seed=seed)
                # wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                # print('stage1_file_path: ' + stage1_file_path)

                # # Log the testing t-SNE plot to wandb
                # stage1_file_path = plot_tsne_delta(
                #     final_model_sep,
                #     X_test_filtered, y_test_filtered, title,
                #     'stage2_testing',
                #     model_type='features_reg',
                #     save_tag=current_time, seed=seed)
                # wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                # print('stage1_file_path: ' + stage1_file_path)

                # # Plot the error histograms
                # filename = plot_error_hist(
                #     final_model_sep,
                #     X_train, y_train,
                #     sample_weights=None,
                #     title=title,
                #     prefix='training')
                # wandb.log({"training_error_hist": wandb.Image(filename)})

                # # Plot the error histograms on the testing set
                # filename = plot_error_hist(
                #     final_model_sep,
                #     X_test, y_test,
                #     sample_weights=None,
                #     title=title,
                #     prefix='testing')
                # wandb.log({"testing_error_hist": wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
