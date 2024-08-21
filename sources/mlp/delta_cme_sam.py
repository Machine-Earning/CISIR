import os
from datetime import datetime

from modules.evaluate.utils import evaluate_mae_pcc, evaluate_plot

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import wandb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback
import numpy as np

from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    get_loss,
    stratified_split,
    set_seed)

from modules.shared.globals import *


def main():
    """
    Main function to run the E-MLP model
    :return:
    """
    for seed in SEEDS:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha in [0.25]:
                    for rho in [0]:  # SAM_RHOS:
                        for add_slope in ADD_SLOPE:
                            # PARAMS
                            outputs_to_use = OUTPUTS_TO_USE

                            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                            # Construct the title
                            title = f'MLP_{inputs_str}_alpha{alpha:.2f}_rho{rho:.2f}_lambda{LAMBDA:.2f}'

                            # Replace any other characters that are not suitable for filenames (if any)
                            title = title.replace(' ', '_').replace(':', '_')

                            # Create a unique experiment name with a timestamp
                            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                            experiment_name = f'{title}_{current_time}'

                            # Set the early stopping patience and learning rate as variables
                            set_seed(seed)
                            patience = PATIENCE  # higher patience
                            learning_rate = START_LR  # og learning rate

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
                            hiddens = MLP_HIDDENS  # hidden layers

                            hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                            loss_key = LOSS_KEY
                            lambda_ = LAMBDA
                            target_change = ('delta_p' in outputs_to_use)
                            alpha_rw = alpha
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

                            # Initialize wandb
                            wandb.init(project="nasa-ts-delta-v7-nl", name=experiment_name, config={
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
                                "loss": loss_key,
                                "lambda": lambda_,
                                "target_change": target_change,
                                "seed": seed,
                                "alpha_rw": alpha_rw,
                                "bandwidth": bandwidth,
                                "reciprocal_reweight": RECIPROCAL_WEIGHTS,
                                "repr_dim": repr_dim,
                                "dropout": dropout,
                                "activation": 'LeakyReLU',
                                "norm": norm,
                                'optimizer': 'adamw',
                                'output_dim': output_dim,
                                'architecture': 'mlp',
                                'cme_speed_threshold': cme_speed_threshold,
                                'residual': residual,
                                'skipped_layers': skipped_layers,
                                'ds_version': DS_VERSION,
                                'mae_plus_th': mae_plus_threshold,
                                'sam_rho': rho
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
                                alpha=COMMON_VAL_ALPHA, bw=bandwidth,
                                min_norm_weight=min_norm_weight,
                                debug=False).reweights
                            print(f'validation set rebalanced.')

                            # get the number of features
                            n_features = X_train.shape[1]
                            print(f'n_features: {n_features}')

                            # create the model
                            model_sep = create_mlp(
                                input_dim=n_features,
                                hiddens=hiddens,
                                repr_dim=repr_dim,
                                output_dim=output_dim,
                                dropout_rate=dropout,
                                activation=activation,
                                norm=norm,
                                residual=residual,
                                skipped_layers=skipped_layers,
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
                                loss={'forecast_head': get_loss(loss_key, lambda_factor=lambda_)},
                            )

                            # Train the model with the callback
                            history = model_sep.fit(
                                X_subtrain,
                                {'forecast_head': y_subtrain},
                                sample_weight=y_subtrain_weights,
                                epochs=epochs, batch_size=batch_size,
                                validation_data=(X_val, {'forecast_head': y_val}, y_val_weights),
                                callbacks=[
                                    early_stopping,
                                    reduce_lr_on_plateau,
                                    WandbCallback(save_model=WANDB_SAVE_MODEL)
                                ],
                                verbose=VERBOSE
                            )

                            # Determine the optimal number of epochs from early stopping
                            # optimal_epochs = early_stopping.best_epoch + 1  # Adjust for the offset

                            # Determine the optimal number of epochs from the fit history
                            optimal_epochs = np.argmin(
                                history.history[ES_CB_MONITOR]) + 1  # +1 to adjust for 0-based index

                            final_model_sep = create_mlp(
                                input_dim=n_features,
                                hiddens=hiddens,
                                repr_dim=repr_dim,
                                output_dim=output_dim,
                                dropout_rate=dropout,
                                activation=activation,
                                norm=norm,
                                residual=residual,
                                skipped_layers=skipped_layers,
                                sam_rho=rho
                            )

                            # Recreate the model architecture
                            final_model_sep.compile(
                                optimizer=AdamW(
                                    learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    beta_1=momentum_beta1
                                ),
                                loss={'forecast_head': get_loss(loss_key, lambda_factor=lambda_)},
                            )

                            # Train on the full dataset
                            final_model_sep.fit(
                                X_train,
                                {'forecast_head': y_train},
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

                            # Evaluate the model on test set
                            evaluate_mae_pcc(
                                final_model_sep, X_test, y_test,
                                above_threshold=mae_plus_threshold,
                                wandb_cls=wandb
                            )

                            # evaluate the model error on train set
                            evaluate_mae_pcc(
                                final_model_sep, X_train, y_train,
                                above_threshold=mae_plus_threshold,
                                wandb_cls=wandb
                            )

                            # Process SEP event files in the specified directory
                            test_directory = root_dir + '/testing'
                            evaluate_plot(
                                final_model_sep,
                                X_test, y_test,
                                lower_threshold=lower_threshold,
                                upper_threshold=upper_threshold,
                                N_samples=N,
                                seed=seed,
                                title=title,
                                model_type='features_reg',
                                test_directory=test_directory,
                                split='test',
                                inputs_to_use=inputs_to_use,
                                outputs_to_use=outputs_to_use,
                                add_slope=add_slope,
                                use_cme=True,
                                cme_speed_threshold=cme_speed_threshold,
                                show_avsp=True,
                                current_time=current_time,
                                wandb_cls=wandb
                            )

                            # training plots
                            test_directory = root_dir + '/training'
                            evaluate_plot(
                                final_model_sep,
                                X_train, y_train,
                                lower_threshold=lower_threshold,
                                upper_threshold=upper_threshold,
                                N_samples=N,
                                seed=seed,
                                title=title,
                                model_type='features_reg',
                                test_directory=test_directory,
                                split='train',
                                inputs_to_use=inputs_to_use,
                                outputs_to_use=outputs_to_use,
                                add_slope=add_slope,
                                use_cme=True,
                                cme_speed_threshold=cme_speed_threshold,
                                show_avsp=True,
                                current_time=current_time,
                                wandb_cls=wandb
                            )

                            # Finish the wandb run
                            wandb.finish()


if __name__ == '__main__':
    main()
