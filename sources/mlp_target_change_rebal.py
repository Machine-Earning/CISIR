import os
from datetime import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
import tensorflow as tf
import numpy as np

from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    evaluate_model,
    process_sep_events,
    get_loss,
    compute_sample_weights,
    PrintBatchMSE)


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:  # , ['e0.5', 'p']]:
        for add_slope in [True]:  # , False]:
            # PARAMS
            # inputs_to_use = ['e0.5']
            # add_slope = True
            
            # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
            inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

            # Construct the title
            title = f'MLP_targetChange_{inputs_str}_slope_{str(add_slope)}_morePlots'

            # Replace any other characters that are not suitable for filenames (if any)
            title = title.replace(' ', '_').replace(':', '_')

            # Create a unique experiment name with a timestamp
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f'{title}_{current_time}'

            # Set the early stopping patience and learning rate as variables
            seed = 123456789
            tf.random.set_seed(seed)
            np.random.seed(seed)
            patience = 3000  # higher patience
            # learning_rate = 3e-5  # og learning rate
            initial_learning_rate = 3e-3
            final_learning_rate = 3e-7
            learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / 3000)
            steps_per_epoch = int(20000 / 8)

            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=steps_per_epoch,
                decay_rate=learning_rate_decay_factor,
                staircase=True)
            
            weight_decay = 0  # higher weight decay
            momentum_beta1 = 0.9  # higher momentum beta1
            batch_size = 8
            epochs = 100000
            hiddens = [100, 100, 50]
            hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
            loss_key = 'mse'
            target_change = True
            # print_batch_mse_cb = PrintBatchMSE()
            rebalacing = True

            # Initialize wandb
            wandb.init(project="mlp-ts-target-change", name=experiment_name, config={
                "inputs_to_use": inputs_to_use,
                "add_slope": add_slope,
                "patience": patience,
                "learning_rate": "lr_decay_3e-3_3e-7",
                "weight_decay": weight_decay,
                "momentum_beta1": momentum_beta1,
                "batch_size": batch_size,
                "epochs": epochs,
                # hidden in a more readable format  (wandb does not support lists)
                "hiddens": hiddens_str,
                "loss": loss_key,
                "target_change": target_change,
                "printing_batch_mse": False,
                "seed": seed,
                "rebalancing": rebalacing
            })

            # set the root directory
            root_dir = 'data/electron_cme_data_split'
            # build the dataset
            X_train, y_train = build_dataset(root_dir + '/training',
                                             inputs_to_use=inputs_to_use,
                                             add_slope=add_slope,
                                             target_change=target_change)
            X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining',
                                                   inputs_to_use=inputs_to_use,
                                                   add_slope=add_slope,
                                                   target_change=target_change)
            X_test, y_test = build_dataset(root_dir + '/testing',
                                           inputs_to_use=inputs_to_use,
                                           add_slope=add_slope,
                                           target_change=target_change)
            X_val, y_val = build_dataset(root_dir + '/validation',
                                         inputs_to_use=inputs_to_use,
                                         add_slope=add_slope,
                                         target_change=target_change)

            # Compute the sample weights
            y_subtrain_weights = compute_sample_weights(y_subtrain)
            y_train_weights = compute_sample_weights(y_train)

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
            mlp_model_sep = create_mlp(input_dim=n_features, hiddens=hiddens)
            mlp_model_sep.summary()

            # Define the EarlyStopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1,
                                           restore_best_weights=True)

            # Compile the model with the specified learning rate
            mlp_model_sep.compile(optimizer=AdamW(learning_rate=learning_rate,
                                                  weight_decay=weight_decay,
                                                  beta_1=momentum_beta1),
                                  loss={'forecast_head': get_loss(loss_key)})

            # Train the model with the callback
            history = mlp_model_sep.fit(X_subtrain,
                                        {'forecast_head': y_subtrain},
                                        sample_weight=y_subtrain_weights,
                                        epochs=epochs, batch_size=batch_size,
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
            optimal_epochs = early_stopping.stopped_epoch - patience + 1  # Adjust for the offset
            final_mlp_model_sep = create_mlp(input_dim=n_features,
                                             hiddens=hiddens)  # Recreate the model architecture
            final_mlp_model_sep.compile(
                optimizer=AdamW(learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                beta_1=momentum_beta1),
                loss={'forecast_head': get_loss(loss_key)})  # Compile the model just like before
            # Train on the full dataset
            final_mlp_model_sep.fit(
                X_train,
                {'forecast_head': y_train},
                sample_weight=y_train_weights,
                epochs=optimal_epochs,
                batch_size=batch_size,
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
                add_slope=add_slope,
                target_change=target_change,
                show_avsp=True)

            # Log the plot to wandb
            for filename in filenames:
                log_title = os.path.basename(filename)
                wandb.log({f'testing_{log_title}': wandb.Image(filename)})

            # Process SEP event files in the specified directory
            test_directory = root_dir + '/training'
            filenames = process_sep_events(
                test_directory,
                final_mlp_model_sep,
                model_type='mlp',
                title=title,
                inputs_to_use=inputs_to_use,
                add_slope=add_slope,
                target_change=target_change,
                show_avsp=True,
                prefix='training')

            # Log the plot to wandb
            for filename in filenames:
                log_title = os.path.basename(filename)
                wandb.log({f'training_{log_title}': wandb.Image(filename)})

            # Finish the wandb run
            wandb.finish()


if __name__ == '__main__':
    main()
