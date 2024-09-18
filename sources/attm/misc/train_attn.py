import os
from datetime import datetime

import numpy as np
import wandb
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.shared.globals import *
from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    get_loss,
    filter_ds,
    stratified_split,
    set_seed)
from sources.attm.modules import create_attentive_model


# Set the environment variable for CUDA (in case it is necessary)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    """
    Main function to run the E-MLP model
    :return:
    """
    for seed in SEEDS:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha, alpha_val in zip([0.5], [1]):
                    for rho in [0.0001]:  # SAM_RHOS:
                        for add_slope in ADD_SLOPE:
                            # PARAMS
                            outputs_to_use = OUTPUTS_TO_USE

                            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                            # Construct the title
                            title = f'ATTM_{inputs_str}_alpha{alpha:.2f}_rho{rho:.2f}_long'

                            # Replace any other characters that are not suitable for filenames (if any)
                            title = title.replace(' ', '_').replace(':', '_')

                            # Create a unique experiment name with a timestamp
                            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                            experiment_name = f'{title}_{current_time}'

                            # Set the early stopping patience and learning rate as variables
                            set_seed(seed)
                            patience = int(12e3) #PATIENCE  # higher patience
                            learning_rate = 1e-5 # og learning rate
                            activation = 'leaky_relu'  # ACTIVATION
                            attn_skipped_layers = 1  # SKIPPED_LAYERS
                            attn_residual = False  # RESIDUAL
                            attn_dropout_rate = 0  # DROPOUT
                            dropout = 0  # DROPOUT
                            attn_norm = None  # NORM
                            norm = 'batch_norm'
                            skipped_blocks = 1  # SKIPPED_LAYERS
                            residual = True  # RESIDUAL

                            reduce_lr_on_plateau = ReduceLROnPlateau(
                                monitor=LR_CB_MONITOR,
                                factor=0.9,
                                patience=500,
                                verbose=VERBOSE,
                                min_delta=LR_CB_MIN_DELTA,
                                min_lr=1e-7)  # LR_CB_MIN_LR)

                            weight_decay = 1e-4  # WEIGHT_DECAY  # higher weight decay
                            momentum_beta1 = MOMENTUM_BETA1  # higher momentum beta1
                            batch_size = BATCH_SIZE  # higher batch size
                            epochs = EPOCHS  # higher epochs
                            # hiddens = MLP_HIDDENS  # hidden layers

                            # hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                            loss_key = 'mse'  # LOSS_KEY
                            target_change = ('delta_p' in outputs_to_use)
                            alpha_rw = alpha
                            bandwidth = BANDWIDTH
                            repr_dim = REPR_DIM
                            output_dim = len(outputs_to_use)
                            # dropout = DROPOUT
                            # activation = ACTIVATION
                            # norm = None  # NORM
                            cme_speed_threshold = cme_speed_threshold
                            # residual = False  # RESIDUAL
                            # skipped_layers = SKIPPED_LAYERS
                            N = N_FILTERED  # number of samples to keep outside the threshold
                            lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                            upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                            mae_plus_threshold = MAE_PLUS_THRESHOLD

                            # Initialize wandb
                            wandb.init(project="train_attn", name=experiment_name, config={
                                "inputs_to_use": inputs_to_use,
                                "add_slope": add_slope,
                                "patience": patience,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "momentum_beta1": momentum_beta1,
                                "batch_size": batch_size,
                                "epochs": epochs,
                                # hidden in a more readable format  (wandb does not support lists)
                                # "hiddens": hiddens_str,
                                "loss": loss_key,
                                "target_change": target_change,
                                "seed": seed,
                                "alpha_rw": alpha_rw,
                                "alpha_val": alpha_val,
                                "bandwidth": bandwidth,
                                "reciprocal_reweight": RECIPROCAL_WEIGHTS,
                                "repr_dim": repr_dim,
                                "dropout": dropout,
                                "activation": 'LeakyReLU',
                                "norm": norm,
                                'optimizer': 'adamw',
                                'output_dim': output_dim,
                                'architecture': 'attm',
                                'cme_speed_threshold': cme_speed_threshold,
                                'residual': residual,
                                'attn_residual': attn_residual,
                                'skipped_blocks': skipped_blocks,
                                'skipped_layers': attn_skipped_layers,
                                'ds_version': DS_VERSION,
                                'mae_plus_th': mae_plus_threshold,
                                'sam_rho': rho,
                                'blocks_hiddens': BLOCKS_HIDDENS,
                                'attn_hiddens': ATTN_HIDDENS
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

                            X_test, y_test = build_dataset(
                                root_dir + '/testing',
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                cme_speed_threshold=cme_speed_threshold)

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

                            X_subtrain, y_subtrain, X_val, y_val = stratified_split(
                                X_train,
                                y_train,
                                shuffle=True,
                                seed=seed,
                                split=VAL_SPLIT,
                                debug=False)

                            # print all cme_files shapes
                            print(f'X_train.shape: {X_train.shape}')
                            print(f'y_train.shape: {y_train.shape}')
                            print(f'X_subtrain.shape: {X_subtrain.shape}')
                            print(f'y_subtrain.shape: {y_subtrain.shape}')
                            print(f'X_test.shape: {X_test.shape}')
                            print(f'y_test.shape: {y_test.shape}')
                            print(f'X_val.shape: {X_val.shape}')
                            print(f'y_val.shape: {y_val.shape}')

                            # Compute the sample weights
                            delta_train = y_train[:, 0]
                            delta_subtrain = y_subtrain[:, 0]
                            delta_val = y_val[:, 0]
                            print(f'delta_train.shape: {delta_train.shape}')
                            print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                            print(f'delta_val.shape: {delta_val.shape}')

                            print(f'rebalancing the training set...')
                            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                            y_train_weights = exDenseReweights(
                                X_train, delta_train,
                                alpha=alpha_rw, bw=bandwidth,
                                min_norm_weight=min_norm_weight,
                                debug=False).reweights
                            print(f'training set rebalanced.')

                            print(f'rebalancing the subtraining set...')
                            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_subtrain)
                            y_subtrain_weights = exDenseReweights(
                                X_subtrain, delta_subtrain,
                                alpha=alpha_rw, bw=bandwidth,
                                min_norm_weight=min_norm_weight,
                                debug=False).reweights
                            print(f'subtraining set rebalanced.')

                            print(f'rebalancing the validation set...')
                            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_val)
                            y_val_weights = exDenseReweights(
                                X_val, delta_val,
                                alpha=alpha_val, bw=bandwidth,
                                min_norm_weight=min_norm_weight,
                                debug=False).reweights
                            print(f'validation set rebalanced.')

                            # get the number of features
                            n_features = X_train.shape[1]
                            print(f'n_features: {n_features}')

                            # create the model
                            model_sep = create_attentive_model(
                                input_dim=n_features,
                                output_dim=output_dim,
                                hidden_blocks=BLOCKS_HIDDENS,
                                attn_hidden_units=ATTN_HIDDENS,
                                attn_hidden_activation=activation,
                                attn_skipped_layers=attn_skipped_layers,
                                attn_residual=attn_residual,
                                attn_dropout_rate=attn_dropout_rate,
                                attn_norm=attn_norm,
                                skipped_blocks=skipped_blocks,
                                repr_dim=repr_dim,
                                dropout_rate=dropout,
                                activation=activation,
                                norm=norm,
                                residual=residual,
                                sam_rho=rho
                            )
                            model_sep.summary()

                            # Define the EarlyStopping callback
                            early_stopping = EarlyStopping(
                                monitor=ES_CB_MONITOR,
                                patience=patience,
                                verbose=VERBOSE,
                                restore_best_weights=ES_CB_RESTORE_WEIGHTS)

                            # Compile the model with the specified learning rate
                            model_sep.compile(
                                optimizer=AdamW(
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    beta_1=momentum_beta1
                                ),
                                loss={'output': get_loss(loss_key)}
                            )

                            # Train the model with the callback
                            history = model_sep.fit(
                                X_subtrain,
                                {'output': y_subtrain},
                                sample_weight=y_subtrain_weights,
                                epochs=epochs, batch_size=batch_size,
                                validation_data=(X_val, {'output': y_val}, y_val_weights),
                                callbacks=[
                                    early_stopping,
                                    reduce_lr_on_plateau,
                                    WandbCallback(save_model=WANDB_SAVE_MODEL)
                                ],
                                verbose=VERBOSE
                            )

                            # Determine the optimal number of epochs from the fit history
                            optimal_epochs = np.argmin(
                                history.history[ES_CB_MONITOR]) + 1  # +1 to adjust for 0-based index
                            # optimal_epochs = int(3e4)

                            final_model_sep = create_attentive_model(
                                input_dim=n_features,
                                output_dim=output_dim,
                                hidden_blocks=BLOCKS_HIDDENS,
                                attn_hidden_units=ATTN_HIDDENS,
                                attn_hidden_activation=activation,
                                attn_skipped_layers=attn_skipped_layers,
                                attn_residual=attn_residual,
                                attn_dropout_rate=attn_dropout_rate,
                                attn_norm=attn_norm,
                                skipped_blocks=skipped_blocks,
                                repr_dim=repr_dim,
                                dropout_rate=dropout,
                                activation=activation,
                                norm=norm,
                                residual=residual,
                                sam_rho=rho
                            )

                            # final_model_sep.summary()
                            # Recreate the model architecture
                            final_model_sep.compile(
                                optimizer=AdamW(
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    beta_1=momentum_beta1
                                ),
                                loss={'output': get_loss(loss_key)}
                            )

                            # Train on the full dataset
                            final_model_sep.fit(
                                X_train,
                                {'output': y_train},
                                sample_weight=y_train_weights,
                                epochs=optimal_epochs,
                                batch_size=batch_size,
                                callbacks=[
                                    reduce_lr_on_plateau,
                                    WandbCallback(save_model=WANDB_SAVE_MODEL)
                                ],
                                verbose=VERBOSE
                            )

                            # Save the final model
                            final_model_sep.save_weights(f"final_model_weights_{experiment_name}_reg.h5")
                            # print where the model weights are saved
                            print(f"Model weights are saved in final_model_weights_{experiment_name}_reg.h5")

                            # evaluate the model error on test set
                            error_mae = evaluate_mae(final_model_sep, X_test, y_test, use_dict=True)
                            print(f'mae error: {error_mae}')
                            wandb.log({"mae": error_mae})

                            # evaluate the model error on training set
                            error_mae_train = evaluate_mae(final_model_sep, X_train, y_train, use_dict=True)
                            print(f'mae error train: {error_mae_train}')
                            wandb.log({"train_mae": error_mae_train})

                            # evaluate the model correlation on test set
                            error_pcc = evaluate_pcc(final_model_sep, X_test, y_test, use_dict=True)
                            print(f'pcc error: {error_pcc}')
                            wandb.log({"pcc": error_pcc})

                            # evaluate the model correlation on training set
                            error_pcc_train = evaluate_pcc(final_model_sep, X_train, y_train, use_dict=True)
                            print(f'pcc error train: {error_pcc_train}')
                            wandb.log({"train_pcc": error_pcc_train})

                            # evaluate the model on test cme_files
                            above_threshold = mae_plus_threshold
                            # evaluate the model error for rare samples on test set
                            error_mae_cond = evaluate_mae(
                                final_model_sep, X_test, y_test, above_threshold=above_threshold, use_dict=True)
                            print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
                            wandb.log({"mae+": error_mae_cond})

                            # evaluate the model error for rare samples on training set
                            error_mae_cond_train = evaluate_mae(
                                final_model_sep, X_train, y_train, above_threshold=above_threshold, use_dict=True)
                            print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                            wandb.log({"train_mae+": error_mae_cond_train})

                            # evaluate the model correlation for rare samples on test set
                            error_pcc_cond = evaluate_pcc(
                                final_model_sep, X_test, y_test, above_threshold=above_threshold, use_dict=True)
                            print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                            wandb.log({"pcc+": error_pcc_cond})

                            # evaluate the model correlation for rare samples on training set
                            error_pcc_cond_train = evaluate_pcc(
                                final_model_sep, X_train, y_train, above_threshold=above_threshold, use_dict=True)
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
                                cme_speed_threshold=cme_speed_threshold,
                                use_dict=True)

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
                                cme_speed_threshold=cme_speed_threshold,
                                use_dict=True)

                            # Log the plot to wandb
                            for filename in filenames:
                                log_title = os.path.basename(filename)
                                wandb.log({f'training_{log_title}': wandb.Image(filename)})

                            # # Evaluate the model correlation with colored
                            # file_path = plot_repr_corr_dist(
                            #     final_model_sep,
                            #     X_train_filtered, y_train_filtered,
                            #     title + "_training",
                            #     model_type='features_reg'
                            # )
                            # wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                            # print('file_path: ' + file_path)
                            #
                            # file_path = plot_repr_corr_dist(
                            #     final_model_sep,
                            #     X_test_filtered, y_test_filtered,
                            #     title + "_test",
                            #     model_type='features_reg'
                            # )
                            # wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                            # print('file_path: ' + file_path)
                            #
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
                            #
                            # # Log the testing t-SNE plot to wandb
                            # stage1_file_path = plot_tsne_delta(
                            #     final_model_sep,
                            #     X_test_filtered, y_test_filtered, title,
                            #     'stage2_testing',
                            #     model_type='features_reg',
                            #     save_tag=current_time, seed=seed)
                            # wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                            # print('stage1_file_path: ' + stage1_file_path)
                            #
                            # # Plot the error histograms
                            # filename = plot_error_hist(
                            #     final_model_sep,
                            #     X_train, y_train,
                            #     sample_weights=None,
                            #     title=title,
                            #     prefix='training')
                            # wandb.log({"training_error_hist": wandb.Image(filename)})
                            #
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
