import os
import random
from datetime import datetime

from modules.training.cme_modeling import pds_space_norm

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # left is 1

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.integration.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_delta, plot_repr_correlation, plot_repr_corr_dist, plot_repr_corr_density, evaluate_pcc
from modules.training import cme_modeling
from modules.training.ts_modeling import build_dataset, create_mlp, filter_ds, stratified_split
from modules.reweighting.exDenseReweightsD import exDenseReweightsD

from modules.shared.globals import *


def main():
    """
    Main function to run the PDS model
    :return:
    """
    # list the devices available
    devices = tf.config.list_physical_devices('GPU')
    print(f'devices: {devices}')
    # Define the dataset options, including the sharding policy

    for SEED in SEEDS:
        mb = cme_modeling.ModelBuilder()
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for add_slope in ADD_SLOPE:
                    # for alpha in [0, 0.1, 0.3, 0.4]:
                    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        # PARAMS
                        # Set NumPy seed
                        np.random.seed(SEED)
                        # Set TensorFlow seed
                        tf.random.set_seed(SEED)
                        # Set random seed
                        random.seed(SEED)
                        # add_slope = True
                        outputs_to_use = OUTPUTS_TO_USE

                        bs = BATCH_SIZE  # full dataset used
                        print(f'batch size : {bs}')

                        # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                        inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                        # Construct the title
                        title = f'MLP_{inputs_str}_slope{str(add_slope)}_PDS_bs{bs}_alpha{alpha:.2f}_CME{cme_speed_threshold}'
                        # Replace any other characters that are not suitable for filenames (if any)
                        title = title.replace(' ', '_').replace(':', '_')

                        # Create a unique experiment name with a timestamp
                        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                        experiment_name = f'{title}_{current_time}'
                        # Set the early stopping patience and learning rate as variables
                        Options = {
                            'batch_size': bs,  # Assuming batch_size is defined elsewhere
                            'epochs': EPOCHS,  # 35k epochs
                            'patience': PATIENCE,
                            'learning_rate': START_LR,  # initial learning rate
                            'weight_decay': WEIGHT_DECAY_PDS,  # Added weight decay
                            'momentum_beta1': MOMENTUM_BETA1,  # Added momentum beta1
                        }

                        hiddens = MLP_HIDDENS
                        hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                        pds = True
                        target_change = ('delta_p' in outputs_to_use)
                        repr_dim = REPR_DIM
                        dropout_rate = DROPOUT
                        activation = ACTIVATION
                        norm = NORM

                        reduce_lr_on_plateau = ReduceLROnPlateau(
                            monitor=LR_CB_MONITOR,
                            factor=LR_CB_FACTOR,
                            patience=LR_CB_PATIENCE,
                            verbose=VERBOSE,
                            min_delta=LR_CB_MIN_DELTA,
                            min_lr=LR_CB_MIN_LR)

                        bandwidth = BANDWIDTH
                        alpha_rw = alpha
                        residual = RESIDUAL
                        skipped_layers = SKIPPED_LAYERS
                        N = N_FILTERED  # number of samples to keep outside the threshold
                        lower_threshold = LOWER_THRESHOLD  # lower threshold for the delta_p
                        upper_threshold = UPPER_THRESHOLD  # upper threshold for the delta_p
                        mae_plus_threshold = MAE_PLUS_THRESHOLD

                        # Initialize wandb
                        wandb.init(project="nasa-ts-delta-v7-pds", name=experiment_name, config={
                            "inputs_to_use": inputs_to_use,
                            "add_slope": add_slope,
                            "target_change": target_change,
                            "patience": Options['patience'],
                            "learning_rate": Options['learning_rate'],
                            "weight_decay": Options['weight_decay'],
                            "momentum_beta1": Options['momentum_beta1'],
                            "batch_size": Options['batch_size'],
                            "epochs": Options['epochs'],
                            # hidden in a more readable format  (wandb does not support lists)
                            "hiddens": hiddens_str,
                            "pds": pds,
                            "seed": SEED,
                            "stage": 1,
                            "reduce_lr_on_plateau": True,
                            "dropout": dropout_rate,
                            "activation": "LeakyReLU",
                            "norm": norm,
                            "optimizer": "adam",
                            "architecture": "mlp",
                            "alpha": alpha_rw,
                            "bandwidth": bandwidth,
                            "residual": residual,
                            "skipped_layers": skipped_layers,
                            "repr_dim": repr_dim,
                            "ds_version": DS_VERSION,
                            "N_freq": N,
                            "lower_t": lower_threshold,
                            "upper_t": upper_threshold,
                            'mae_plus_th': mae_plus_threshold
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
                            N=N, seed=SEED)

                        X_test_filtered, y_test_filtered = filter_ds(
                            X_test, y_test,
                            low_threshold=lower_threshold,
                            high_threshold=upper_threshold,
                            N=N, seed=SEED)

                        # pds normalize the data
                        y_train_norm, norm_lower_t, norm_upper_t = pds_space_norm(y_train)
                        # y_test_norm = pds_space_norm(y_test)

                        # print all cme_files shapes
                        print(f'X_train.shape: {X_train.shape}')
                        print(f'y_train.shape: {y_train.shape}')
                        print(f'X_test.shape: {X_test.shape}')
                        print(f'y_test.shape: {y_test.shape}')

                        # get the number of features
                        n_features = X_train.shape[1]
                        print(f'n_features: {n_features}')

                        # Compute the sample weights
                        delta_train = y_train_norm[:, 0]
                        print(f'delta_train.shape: {delta_train.shape}')

                        print(f'rebalancing the training set...')
                        min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)

                        train_weights_dict = exDenseReweightsD(
                            X_train, delta_train,
                            alpha=alpha_rw, bw=bandwidth,
                            min_norm_weight=min_norm_weight,
                            debug=False).label_reweight_dict
                        print(f'done rebalancing the training set...')

                        # get subtrain and val
                        X_subtrain, y_subtrain_norm, X_val, y_val_norm = stratified_split(
                            X_train,
                            y_train_norm,
                            shuffle=True,
                            seed=SEED,
                            split=VAL_SPLIT,
                            debug=False)

                        # filter validation set
                        # X_val, y_val_norm = filter_ds(
                        #     X_val, y_val_norm,
                        #     low_threshold=norm_lower_t,
                        #     high_threshold=norm_upper_t,
                        #     N=200, seed=SEED)

                        print(f'done rebalancing the subtraining set...')
                        print(f'X_val.shape: {X_val.shape}')
                        print(f'y_val.shape: {y_val_norm.shape}')

                        # get the number of features
                        n_features = X_train.shape[1]
                        print(f'n_features: {n_features}')

                        # create the model
                        model_sep = create_mlp(
                            input_dim=n_features,
                            hiddens=hiddens,
                            output_dim=0,
                            pds=pds,
                            repr_dim=repr_dim,
                            dropout_rate=dropout_rate,
                            activation=activation,
                            norm=norm,
                            residual=residual,
                            skipped_layers=skipped_layers
                        )
                        model_sep.summary()

                        mb.train_pds(
                            model_sep,
                            X_train, y_train,
                            X_subtrain, y_subtrain_norm,
                            X_val, y_val_norm,
                            train_label_weights_dict=train_weights_dict,
                            learning_rate=Options['learning_rate'],
                            epochs=Options['epochs'],
                            batch_size=Options['batch_size'],
                            patience=Options['patience'],
                            save_tag=current_time + title + "_features_noinj",
                            callbacks_list=[
                                WandbCallback(save_model=False),
                                reduce_lr_on_plateau
                            ]
                        )

                        # evaluate the model on test cme_files
                        above_threshold = mae_plus_threshold
                        error_pcc_cond = evaluate_pcc(
                            model_sep, X_test, y_test, i_above_threshold=above_threshold)

                        print(f'pcc error delta i>= 0.5 test: {error_pcc_cond}')
                        # Log the MAE error to wandb
                        wandb.log({"pcc_error_cond_test": error_pcc_cond})

                        error_pcc = evaluate_pcc(model_sep, X_test, y_test)

                        print(f'pcc error delta test: {error_pcc}')
                        # Log the MAE error to wandb
                        wandb.log({"pcc_error_test": error_pcc})

                        # Evaluate the model correlation with colored
                        file_path = plot_repr_corr_dist(
                            model_sep,
                            X_train_filtered, y_train_filtered,
                            title + "_training"
                        )
                        wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        file_path = plot_repr_corr_dist(
                            model_sep,
                            X_test_filtered, y_test_filtered,
                            title + "_test"
                        )
                        wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        # Log t-SNE plot
                        # Log the training t-SNE plot to wandb
                        stage1_file_path = plot_tsne_delta(
                            model_sep,
                            X_train_filtered, y_train_filtered, title,
                            'stage1_training',
                            model_type='features',
                            save_tag=current_time, seed=SEED)
                        wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                        print('stage1_file_path: ' + stage1_file_path)

                        # Log the testing t-SNE plot to wandb
                        stage1_file_path = plot_tsne_delta(
                            model_sep,
                            X_test_filtered, y_test_filtered, title,
                            'stage1_testing',
                            model_type='features',
                            save_tag=current_time, seed=SEED)
                        wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                        print('stage1_file_path: ' + stage1_file_path)

                        # Evaluate the model correlation
                        file_path = plot_repr_correlation(
                            model_sep,
                            X_train_filtered, y_train_filtered,
                            title + "_training"
                        )
                        wandb.log({'representation_correlation_plot_train': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        file_path = plot_repr_correlation(
                            model_sep,
                            X_test_filtered, y_test_filtered,
                            title + "_test"
                        )
                        wandb.log({'representation_correlation_plot_test': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        # Evaluate the model correlation density
                        file_path = plot_repr_corr_density(
                            model_sep,
                            X_train_filtered, y_train_filtered,
                            title + "_training"
                        )
                        wandb.log({'representation_correlation_density_plot_train': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        file_path = plot_repr_corr_density(
                            model_sep,
                            X_test_filtered, y_test_filtered,
                            title + "_test"
                        )
                        wandb.log({'representation_correlation_density_plot_test': wandb.Image(file_path)})
                        print('file_path: ' + file_path)

                        # Finish the wandb run
                        wandb.finish()


if __name__ == '__main__':
    main()
