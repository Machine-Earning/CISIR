import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_pds, plot_repr_correlation
from modules.training import cme_modeling
from modules.training.ts_modeling import build_dataset, create_mlp

# SEEDING
SEED = 42  # seed number

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
        for add_slope in [False]:
            # PARAMS
            # inputs_to_use = ['e0.5']
            # add_slope = True
            bs = 2000  # full dataset used
            print(f'batch size : {bs}')

            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

            # Construct the title
            title = f'MLP_{inputs_str}_slope_{str(add_slope)}_PDS_bs_{bs}'

            # Replace any other characters that are not suitable for filenames (if any)
            title = title.replace(' ', '_').replace(':', '_')

            # Create a unique experiment name with a timestamp
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f'{title}_{current_time}'
            # Set the early stopping patience and learning rate as variables
            Options = {
                'batch_size': bs,  # Assuming batch_size is defined elsewhere
                'epochs': 10000,
                'patience': 1000,  # Updated to 50
                'learning_rate': 1e-3,  # Updated to 3e-4
                'weight_decay': 0,  # Added weight decay
                'momentum_beta1': 0.95,  # Added momentum beta1
            }
            hiddens = [100, 100, 50]
            hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
            pds = True
            # Callback for reducing learning rate when a metric has stopped improving
            reduce_lr_on_plateau_cb = ReduceLROnPlateau(
                monitor='val_loss',  # Metric to monitor
                factor=0.3,  # Factor by which the learning rate will be reduced. new_lr = lr * factor
                patience=150,  # Number of epochs with no improvement after which learning rate will be reduced
                verbose=1,  # If 1, prints a message when reducing the learning rate
                mode='min',  # In 'min' mode, lr will reduce when the quantity monitored has stopped decreasing
                min_delta=1e-4,  # Threshold for measuring the new optimum, to only focus on significant changes
                cooldown=1,  # Number of epochs to wait before resuming normal operation after lr has been reduced
                min_lr=1e-8  # Lower bound on the learning rate
            )

            # Initialize wandb
            wandb.init(project="mlp-ts-pds", name=experiment_name, config={
                "inputs_to_use": inputs_to_use,
                "add_slope": add_slope,
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
                "reduce_lr_on_plateau": True
            })

            # set the root directory
            # root_dir = 'D:/College/Fall2023/electron_cme_v5/electron_cme_data_split'
            root_dir = "data/electron_cme_data_split"
            # build the dataset
            X_train, y_train = build_dataset(root_dir + '/training', inputs_to_use=inputs_to_use, add_slope=add_slope)
            X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining', inputs_to_use=inputs_to_use,
                                                   add_slope=add_slope)
            X_test, y_test = build_dataset(root_dir + '/testing', inputs_to_use=inputs_to_use, add_slope=add_slope)
            X_val, y_val = build_dataset(root_dir + '/validation', inputs_to_use=inputs_to_use, add_slope=add_slope)

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
            n_features = X_train.shape[1]
            print(f'n_features: {n_features}')

            # create the model
            mlp_model_sep = create_mlp(input_dim=n_features, hiddens=hiddens, output_dim=0, pds=pds)
            mlp_model_sep.summary()

            mb.train_pds(mlp_model_sep,
                         X_subtrain, y_subtrain,
                         X_val, y_val,
                         X_train, y_train,
                         learning_rate=Options['learning_rate'],
                         epochs=Options['epochs'],
                         batch_size=Options['batch_size'],
                         patience=Options['patience'], save_tag=current_time + "_features",
                         callbacks_list=[WandbCallback(), reduce_lr_on_plateau_cb])

            file_path = plot_tsne_pds(mlp_model_sep,
                                      X_train,
                                      y_train,
                                      title, 'training',
                                      save_tag=current_time)

            # Log t-SNE plot for training
            # Log the training t-SNE plot to wandb
            wandb.log({'tsne_training_plot': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_tsne_pds(mlp_model_sep,
                                      X_test,
                                      y_test,
                                      title, 'testing',
                                      save_tag=current_time)

            # Log t-SNE plot for testing
            # Log the testing t-SNE plot to wandb
            wandb.log({'tsne_testing_plot': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            file_path = plot_repr_correlation(mlp_model_sep, X_test, y_test, title)
            # Log the representation correlation plot to wandb
            wandb.log({'representation_correlation_plot': wandb.Image(file_path)})
            print('file_path: ' + file_path)

            # Finish the wandb run
            wandb.finish()


if __name__ == '__main__':
    main()
