from electron_modeling import build_dataset, create_mlp, evaluate_model, process_sep_events
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    # build the dataset
    X_train, y_train = build_dataset('D:/College/Fall2023/electron_cme_v4/electron_cme_data_split/training')
    X_subtrain, y_subtrain = build_dataset('D:/College/Fall2023/electron_cme_v4/electron_cme_data_split/subtraining')
    X_test, y_test = build_dataset('D:/College/Fall2023/electron_cme_v4/electron_cme_data_split/testing')
    X_val, y_val = build_dataset('D:/College/Fall2023/electron_cme_v4/electron_cme_data_split/validation')

    # print all data shapes
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    print(f'X_subtrain.shape: {X_subtrain.shape}')
    print(f'y_subtrain.shape: {y_subtrain.shape}')
    print(f'X_test.shape: {X_test.shape}')
    print(f'y_test.shape: {y_test.shape}')
    print(f'X_val.shape: {X_val.shape}')
    print(f'y_val.shape: {y_val.shape}')

    # print a sample of the training data
    print(f'X_train[0]: {X_train[0]}')
    print(f'y_train[0]: {y_train[0]}')

    # create the model
    mlp_model_sep = create_mlp(input_dim=75, hiddens=[100, 100, 50])
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
                                callbacks=[early_stopping])

    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # save the plot
    plt.savefig('mlp_loss.png')

    # Determine the optimal number of epochs from early stopping
    optimal_epochs = early_stopping.stopped_epoch - patience + 1  # Adjust for the offset
    final_mlp_model_sep = create_mlp(input_dim=75, hiddens=[100, 100, 50])  # Recreate the model architecture
    final_mlp_model_sep.compile(optimizer=Adam(learning_rate=learning_rate),
                                loss={'forecast_head': 'mse'})  # Compile the model just like before
    # Train on the full dataset
    final_mlp_model_sep.fit(X_train, {'forecast_head': y_train}, epochs=optimal_epochs, batch_size=32, verbose=1)

    # evaluate the model on test data
    error_mae = evaluate_model(final_mlp_model_sep, X_test, y_test)
    print(f'mae error: {error_mae}')

    # Process SEP event files in the specified directory
    test_directory = 'D:/College/Fall2023/electron_cme_v4/electron_cme_data_split/testing'
    title = 'E-MLP'
    process_sep_events(test_directory, final_mlp_model_sep, using_y_model=False, title=title)


if __name__ == '__main__':
    main()
