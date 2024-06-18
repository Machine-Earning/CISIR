import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback

from modules.dataload.ts_modeling import build_dataset, create_mlp, evaluate_model, process_sep_events

# Seeds for reproducibility
seed_value = 42


def set_seeds(seed_value):
    """
    Set the random seeds for reproducibility
    :param seed_value:  The seed value to set
    :return:        None
    """
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)


def weighted_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor, gamma: float, epsilon: float = 1e-20) -> tf.Tensor:
    """
    Compute the Mean Squared Error weighted by a factor of (target_value / divisor)^gamma.

    Parameters:
    - y_true (tf.Tensor): True values.
    - y_pred (tf.Tensor): Predicted values.
    - gamma (float): Exponent to which the weights are raised.
    - epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-6.

    Returns:
    - tf.Tensor: The computed weighted MSE loss.
    """
    divisor = 5.86  # The maximum value of the target variable in the training set
    # changing it to 4 would emphasize higher values more
    # divisor = 4
    # Calculate the weights (target_value / divisor)^gamma for each instance
    # Adding epsilon to avoid division by zero
    weights = tf.pow((y_true / divisor) + epsilon, gamma)

    # Calculate MSE
    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

    # Apply the weights
    weighted_mse = weights * mse

    # Return the mean of weighted MSE
    return tf.reduce_mean(weighted_mse)


def main():
    """
    Main function to run the E-MLP model
    :return:
    """
    gammas = [.7744]
    for gamma in gammas:
        for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
            for add_slope in [False]:
                # PARAMS
                # inputs_to_use = ['e0.5']
                # add_slope = True
                set_seeds(seed_value)
                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                # Construct the title
                title = f'MLP_{inputs_str}_add_slope_{str(add_slope)}_gamma{gamma}'

                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')

                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'

                # Initialize wandb
                wandb.init(project="mlp-ts-gamma", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "gamma": gamma
                })

                # set the root directory
                root_dir = 'D:/College/Fall2023/electron_cme_v4/electron_cme_data_split'
                # build the dataset
                X_train, y_train = build_dataset(root_dir + '/training', inputs_to_use=inputs_to_use,
                                                 add_slope=add_slope)
                X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining', inputs_to_use=inputs_to_use,
                                                       add_slope=add_slope)
                X_test, y_test = build_dataset(root_dir + '/testing', inputs_to_use=inputs_to_use, add_slope=add_slope)
                X_val, y_val = build_dataset(root_dir + '/validation', inputs_to_use=inputs_to_use, add_slope=add_slope)

                # print all cme_files shapes
                # print(f'X_train.shape: {X_train.shape}')
                # print(f'y_train.shape: {y_train.shape}')
                # print(f'X_subtrain.shape: {X_subtrain.shape}')
                # print(f'y_subtrain.shape: {y_subtrain.shape}')
                # print(f'X_test.shape: {X_test.shape}')
                # print(f'y_test.shape: {y_test.shape}')
                # print(f'X_val.shape: {X_val.shape}')
                # print(f'y_val.shape: {y_val.shape}')
                # 
                # # print a sample of the training cme_files
                # print(f'X_train[0]: {X_train[0]}')
                # print(f'y_train[0]: {y_train[0]}')

                # get the number of features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')
                hiddens = [100, 100, 50]

                # create the model
                mlp_model_sep = create_mlp(input_dim=n_features, hiddens=hiddens)
                mlp_model_sep.summary()

                # Set the early stopping patience and learning rate as variables
                patience = 50
                weight_decay = 0  # higher weight decay
                momentum_beta1 = 0.9  # higher momentum beta1
                learning_rate = 3e-3  # higher learning rate

                # Define the EarlyStopping callback
                early_stopping = EarlyStopping(monitor='val_forecast_head_loss', patience=patience, verbose=1,
                                               restore_best_weights=True)

                # Compile the model with the specified learning rate
                mlp_model_sep.compile(optimizer=AdamW(learning_rate=learning_rate,
                                                      weight_decay=weight_decay,
                                                      beta_1=momentum_beta1),
                                      loss={'forecast_head':
                                                lambda y_true, y_pred: weighted_mse_loss(y_true, y_pred, gamma)
                                            })

                # Train the model with the callback
                history = mlp_model_sep.fit(X_subtrain,
                                            {'forecast_head': y_subtrain},
                                            epochs=1000, batch_size=32,
                                            validation_data=(X_val, {'forecast_head': y_val}),
                                            callbacks=[early_stopping, WandbCallback()])

                # Plot the training and validation loss
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                # save the plot
                plt.savefig(f'mlp_loss_{title}.png')

                # Determine the optimal number of epochs from early stopping
                optimal_epochs = early_stopping.stopped_epoch  + 1  # Adjust for the offset
                final_mlp_model_sep = create_mlp(input_dim=n_features,
                                                 hiddens=hiddens)  # Recreate the model architecture
                final_mlp_model_sep.compile(optimizer=AdamW(learning_rate=learning_rate,
                                                            weight_decay=weight_decay,
                                                            beta_1=momentum_beta1),
                                            loss={'forecast_head':
                                                      lambda y_true, y_pred: weighted_mse_loss(y_true, y_pred, gamma)
                                                  })
                # Train on the full dataset
                final_mlp_model_sep.fit(X_train, {'forecast_head': y_train}, epochs=optimal_epochs, batch_size=32,
                                        verbose=1)

                # evaluate the model on test cme_files
                error_mae = evaluate_model(final_mlp_model_sep, X_test, y_test)
                print(f'mae error: {error_mae}')
                # Log the MAE error to wandb
                wandb.log({"mae_error": error_mae})

                # Process SEP event files in the specified directory
                test_directory = root_dir + '/testing'
                filenames = process_sep_events(
                    test_directory,
                    final_mlp_model_sep,
                    model_type='mlp',
                    title=title,
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope)

                # Log the plot to wandb
                for filename in filenames:
                    wandb.log({f'{filename}': wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
