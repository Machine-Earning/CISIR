import os
import random
from datetime import datetime

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta, plot_repr_correlation, plot_repr_corr_density

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from modules.training import cme_modeling
from modules.training.cme_modeling import pds_space_norm
from modules.training.ts_modeling import build_dataset, create_mlp, reshape_X, filter_ds

# SEEDING
SEED = 456789  # seed number

# Set NumPy seed
np.random.seed(SEED)

# Set TensorFlow seed
tf.random.set_seed(SEED)

# Set random seed
random.seed(SEED)

mb = cme_modeling.ModelBuilder()


def main():
    """
    Main function to run the PDS model
    :return:
    """
    # list the devices available
    devices = tf.config.list_physical_devices('GPU')
    print(f'devices: {devices}')
    # Define the dataset options, including the sharding policy

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for cme_speed_threshold in [0]:
            for add_slope in [False, True]:
                # PARAMS
                # inputs_to_use = ['e0.5']
                # add_slope = True
                outputs_to_use = ['delta_p']

                bs = 4096  # full dataset used
                print(f'batch size : {bs}')

                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                # Construct the title
                title = f'MLP_{inputs_str}_slope{str(add_slope)}_PDSinj_bs{bs}_CME{cme_speed_threshold}'

                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')

                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'
                # Set the early stopping patience and learning rate as variables
                Options = {
                    'batch_size': bs,  # Assuming batch_size is defined elsewhere
                    'epochs': int(2e4),  # 35k epochs
                    'learning_rate': 1e-2,  # initial learning rate
                    'weight_decay': 1e-8,  # Added weight decay
                    'momentum_beta1': 0.9,  # Added momentum beta1
                }
                hiddens = [
                    2048, 1024,
                    2048, 1024,
                    1024, 512,
                    1024, 512,
                    512, 256,
                    512, 256,
                    256, 128,
                    256, 128,
                    256, 128,
                    128, 128,
                    128, 128,
                    128, 128
                ]
                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                pds = True
                target_change = ('delta_p' in outputs_to_use)
                repr_dim = 128
                dropout_rate = 0.5
                activation = None
                norm = 'batch_norm'
                reduce_lr_on_plateau = ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=1000,
                    verbose=1,
                    min_delta=1e-5,
                    min_lr=1e-10)
                residual = True
                skipped_layers = 2
                N = 200  # number of samples to keep outside the threshold
                lower_threshold = -0.5  # lower threshold for the delta_p
                upper_threshold = 0.5  # upper threshold for the delta_p

                # Initialize wandb
                wandb.init(project="nasa-ts-delta-overfit", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "target_change": target_change,
                    # "patience": Options['patience'],
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
                    'cme_speed_threshold': cme_speed_threshold,
                    "residual": residual,
                    "skipped_layers": skipped_layers,
                    "repr_dim": repr_dim,
                    "ds_version": 5,
                    "N_freq": N,
                    "lower_t": lower_threshold,
                    "upper_t": upper_threshold
                })

                # set the root directory
                root_dir = "data/electron_cme_data_split_v5"
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

                # print a sample of the training cme_files
                # print(f'X_train[0]: {X_train[0]}')
                # print(f'y_train[0]: {y_train[0]}')

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

                print('Reshaping input for model')
                X_train = reshape_X(
                    X_train,
                    [n_features],
                    inputs_to_use,
                    add_slope,
                    model_sep.name)

                # X_test = reshape_X(
                #     X_test,
                #     [n_features],
                #     inputs_to_use,
                #     add_slope,
                #     model_sep.name)

                mb.overtrain_pds_inj(
                    model_sep,
                    X_train, y_train_norm,
                    learning_rate=Options['learning_rate'],
                    epochs=Options['epochs'],
                    batch_size=Options['batch_size'],
                    save_tag=current_time + title + "_features_128_sl",
                    lower_bound=norm_lower_t,
                    upper_bound=norm_upper_t,
                    callbacks_list=[
                        WandbCallback(save_model=False),
                        reduce_lr_on_plateau
                    ]
                )

                ##Evalute the model correlation with colored
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

                ## Log t-SNE plot
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

                ## Evalute the model correlation
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

                ## Evalute the model correlation density
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
