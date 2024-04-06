import os
import random
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_delta, plot_repr_correlation
from modules.training import cme_modeling
from modules.training.ts_modeling import build_dataset, create_1dcnn, reshape_X

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
        for add_slope in [True, False]:
            # PARAMS
            # inputs_to_use = ['e0.5']
            # add_slope = True
            outputs_to_use = ['delta_p']

            bs = 5000  # full dataset used
            print(f'batch size : {bs}')

            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

            # Construct the title
            title = f'1DCNN_{inputs_str}_slope{str(add_slope)}_PDS_bs{bs}'

            # Replace any other characters that are not suitable for filenames (if any)
            title = title.replace(' ', '_').replace(':', '_')

            # Create a unique experiment name with a timestamp
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f'{title}_{current_time}'
            # Set the early stopping patience and learning rate as variables
            Options = {
                'batch_size': bs,  # Assuming batch_size is defined elsewhere
                'epochs': 50000,
                'patience': 5000,  # Updated to 50
                'learning_rate': 1e-1,  # Updated to 3e-4
                'weight_decay': 1e-8,  # Added weight decay
                'momentum_beta1': 0.97,  # Added momentum beta1
            }
            hiddens = [
                (32, 10, 1, 'none', 0),  # Conv1: Start with broad features
                (64, 8, 1, 'max', 2),  # Conv2 + Pool: Start to reduce and capture features
                (64, 7, 1, 'none', 0),  # Conv3: Further detail without reducing dimension
                (128, 5, 1, 'none', 0),  # Conv4: Increase filters, capture more refined features
                (128, 5, 1, 'max', 2),  # Conv5 + Pool: Reduce dimension and increase depth
                (256, 3, 1, 'none', 0),  # Conv6: Increase depth without immediate pooling
                (256, 3, 1, 'none', 0),  # Conv7: Continue with high capacity, no dimension reduction
                (512, 3, 1, 'max', 2),  # Conv8 + Pool: Use max pooling for stronger feature selection
                (512, 3, 1, 'none', 0),  # Conv9: Maintain capacity, focusing on detailed features
                # Note: Assuming a reduction towards dense layers after Conv9
            ]
            hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
            pds = True
            target_change = ('delta_p' in outputs_to_use)
            repr_dim = 9
            dropout_rate = 0.5
            activation = None
            norm = 'batch_norm'
            reduce_lr_on_plateau = ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=300,
                verbose=1,
                min_delta=1e-5,
                min_lr=1e-10)

            # Initialize wandb
            wandb.init(project="nasa-ts-pds-delta", name=experiment_name, config={
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
                "optimizer": "adamw",
                "architecture": "1dcnn",
            })

            # set the root directory
            root_dir = "data/electron_cme_data_split"
            # build the dataset
            X_train, y_train = build_dataset(root_dir + '/training',
                                             inputs_to_use=inputs_to_use,
                                             add_slope=add_slope,
                                             outputs_to_use=outputs_to_use)
            X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining',
                                                   inputs_to_use=inputs_to_use,
                                                   add_slope=add_slope,
                                                   outputs_to_use=outputs_to_use)
            X_test, y_test = build_dataset(root_dir + '/testing',
                                           inputs_to_use=inputs_to_use,
                                           add_slope=add_slope,
                                           outputs_to_use=outputs_to_use)
            X_val, y_val = build_dataset(root_dir + '/validation',
                                         inputs_to_use=inputs_to_use,
                                         add_slope=add_slope,
                                         outputs_to_use=outputs_to_use)

            # print all cme_files shapes
            print(f'X_train.shape: {X_train.shape}')
            print(f'y_train.shape: {y_train.shape}')
            print(f'X_subtrain.shape: {X_subtrain.shape}')
            print(f'y_subtrain.shape: {y_subtrain.shape}')
            print(f'X_test.shape: {X_test.shape}')
            print(f'y_test.shape: {y_test.shape}')
            print(f'X_val.shape: {X_val.shape}')
            print(f'y_val.shape: {y_val.shape}')

            # print a sample of the training cme_files
            # print(f'X_train[0]: {X_train[0]}')
            # print(f'y_train[0]: {y_train[0]}')

            # get the number of features
            # n_features = X_train.shape[1]
            # print(f'n_features: {n_features}')
            # get the number of features
            if add_slope:
                # n_features = [25] * len(inputs_to_use) * 2
                n_features = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
            else:
                n_features = [25] * len(inputs_to_use)
            print(f'n_features: {n_features}')

            # create the model
            mlp_model_sep = create_1dcnn(
                input_dims=n_features,
                hiddens=hiddens,
                output_dim=0,
                pds=pds,
                repr_dim=repr_dim,
                dropout_rate=dropout_rate,
                activation=activation,
                norm=norm
            )
            mlp_model_sep.summary()

            print('Reshaping input for model')
            X_subtrain = reshape_X(
                X_subtrain,
                n_features,
                inputs_to_use,
                add_slope,
                mlp_model_sep.name)

            X_val = reshape_X(
                X_val,
                n_features,
                inputs_to_use,
                add_slope,
                mlp_model_sep.name)

            X_train = reshape_X(
                X_train,
                n_features,
                inputs_to_use,
                add_slope,
                mlp_model_sep.name)

            X_test = reshape_X(
                X_test,
                n_features,
                inputs_to_use,
                add_slope,
                mlp_model_sep.name)

            mb.train_pds(mlp_model_sep,
                         X_subtrain, y_subtrain,
                         X_val, y_val,
                         X_train, y_train,
                         learning_rate=Options['learning_rate'],
                         epochs=Options['epochs'],
                         batch_size=Options['batch_size'],
                         patience=Options['patience'], save_tag=current_time + title + "_features",
                         callbacks_list=[WandbCallback(save_model=False), reduce_lr_on_plateau])

            file_path = plot_tsne_delta(mlp_model_sep, X_train, y_train, title, 'training', save_tag=current_time,
                                        seed=SEED)

            # Log t-SNE plot for training
            # Log the training t-SNE plot to wandb
            wandb.log({'tsne_training_plot': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_tsne_delta(mlp_model_sep, X_test, y_test, title, 'testing', save_tag=current_time,
                                        seed=SEED)

            # Log t-SNE plot for testing
            # Log the testing t-SNE plot to wandb
            wandb.log({'tsne_testing_plot': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_repr_correlation(mlp_model_sep, X_val, y_val, title + "_training")
            # Log the representation correlation plot to wandb
            wandb.log({'representation_correlation_plot_train': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_repr_correlation(mlp_model_sep, X_test, y_test, title + "_test")
            # Log the representation correlation plot to wandb
            wandb.log({'representation_correlation_plot_test': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            # Finish the wandb run
            wandb.finish()


if __name__ == '__main__':
    main()
