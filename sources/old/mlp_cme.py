from modules.dataload.ts_modeling import build_full_dataset, create_mlp, evaluate_model, process_sep_events
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import wandb
from datetime import datetime
from wandb.keras import WandbCallback


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8'], ['e0.5', 'e1.8', 'p']]:
        for add_slope in [True, False]:
            for cme_speed_threshold in [0, 500]:
                # PARAMS
                # inputs_to_use = ['e0.5']
                # add_slope = True

                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                # Construct the title
                title = f'MLP_withCME_{inputs_str}_add_slope_{str(add_slope)}_cme_speed_{cme_speed_threshold}'

                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')

                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'

                # Initialize wandb
                wandb.init(project="mlp-ts-cme", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                })

                # set the root directory
                root_dir = 'D:/College/Fall2023/electron_cme_v4/electron_cme_data_split'
                # build the dataset
                X_train, y_train = build_full_dataset(root_dir + '/training', inputs_to_use=inputs_to_use,
                                                      add_slope=add_slope, cme_speed_threshold=cme_speed_threshold)
                X_subtrain, y_subtrain = build_full_dataset(root_dir + '/subtraining', inputs_to_use=inputs_to_use,
                                                            add_slope=add_slope,
                                                            cme_speed_threshold=cme_speed_threshold)
                X_test, y_test = build_full_dataset(root_dir + '/testing', inputs_to_use=inputs_to_use,
                                                    add_slope=add_slope, cme_speed_threshold=cme_speed_threshold)
                X_val, y_val = build_full_dataset(root_dir + '/validation', inputs_to_use=inputs_to_use,
                                                  add_slope=add_slope, cme_speed_threshold=cme_speed_threshold)

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
                print(f'X_train[0]: {X_train[0]}')
                print(f'y_train[0]: {y_train[0]}')

                # get the number of features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')
                hiddens = [100, 100, 50]

                # create the model
                mlp_model_sep = create_mlp(input_dim=n_features, hiddens=hiddens)
                mlp_model_sep.summary()

                # Set the early stopping patience and learning rate as variables
                patience = 50
                learning_rate = 3e-3

                # Define the EarlyStopping callback
                early_stopping = EarlyStopping(monitor='val_forecast_head_loss', patience=patience, verbose=1,
                                               restore_best_weights=True)

                # Compile the model with the specified learning rate
                mlp_model_sep.compile(optimizer=Adam(learning_rate=learning_rate), loss={'forecast_head': 'mse'})

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
                final_mlp_model_sep.compile(optimizer=Adam(learning_rate=learning_rate),
                                            loss={'forecast_head': 'mse'})  # Compile the model just like before
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
                    using_cme=True,
                    title=title,
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    cme_speed_threshold=cme_speed_threshold)

                # Log the plot to wandb
                for filename in filenames:
                    wandb.log({f'{filename}': wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
