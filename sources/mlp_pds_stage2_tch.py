from datetime import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_pds, plot_tsne_extended
from modules.training.cme_modeling import ModelBuilder
from modules.training.ts_modeling import (build_dataset,
                                          create_mlp,
                                          evaluate_model,
                                          process_sep_events,
                                          get_loss)

mb = ModelBuilder()


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for add_slope in [True]:
            for freeze in [False, True]:
                # PARAMS
                # inputs_to_use = ['e0.5']
                # add_slope = True

                # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                # Construct the title
                title = f'MLP_PDS_Stage2_{inputs_str}_slope_{str(add_slope)}_frozen{freeze}'

                # Replace any other characters that are not suitable for filenames (if any)
                title = title.replace(' ', '_').replace(':', '_')

                # Create a unique experiment name with a timestamp
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'

                seed = 42
                hiddens = [100, 100, 50]
                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                # Set the early stopping patience and learning rate as variables
                patience = 1000
                learning_rate = 3e-4
                weight_decay = 0  # higher weight decay
                momentum_beta1 = 0.9  # higher momentum beta1
                batch_size = 32
                epochs = 100000
                loss_key = 'mse'
                tf.random.set_seed(seed)
                target_change = True
                pds = True
                # weight_path = '/home1/jmoukpe2016/keras-functional-api/model_weights_20240205-230529_features.h5'
                # weight_path = '/home1/jmoukpe2016/keras-functional-api/model_weights_20240222-192939_features.h5'
                weight_path = '/home1/jmoukpe2016/keras-functional-api/model_weights_20240223-000002_features.h5'

                # Initialize wandb
                wandb.init(project="mlp-ts-pds", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "freeze": freeze,
                    "hiddens": hiddens_str,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "momentum_beta1": momentum_beta1,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "seed": seed,
                    "target_change": target_change,
                    "pds": pds,
                    "stage": 2,
                    "stage1_weights": weight_path
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
                mlp_model_sep_stage1 = create_mlp(input_dim=n_features, hiddens=hiddens, output_dim=0, pds=pds)
                mlp_model_sep_stage1.summary()
                # load the weights from the first stage

                mlp_model_sep_stage1.load_weights(weight_path)
                # print the save 
                print(f'weights loaded successfully from: {weight_path}')

                # Log t-SNE plot for training
                # Log the training t-SNE plot to wandb
                stage1_file_path = plot_tsne_pds(mlp_model_sep_stage1,
                                                 X_train,
                                                 y_train,
                                                 title, 'stage1_training',
                                                 save_tag=current_time)
                wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                # Log t-SNE plot for testing
                # Log the testing t-SNE plot to wandb
                stage1_file_path = plot_tsne_pds(mlp_model_sep_stage1,
                                                 X_test,
                                                 y_test,
                                                 title, 'stage1_testing',
                                                 save_tag=current_time)
                wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                print('stage1_file_path: ' + stage1_file_path)

                mlp_model_sep = mb.add_proj_head(mlp_model_sep_stage1, output_dim=1, freeze_features=freeze, pds=pds)
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

                final_mlp_model_sep_stage1 = create_mlp(input_dim=n_features, hiddens=hiddens, output_dim=0, pds=pds)
                final_mlp_model_sep_stage1.load_weights(weight_path)

                # Recreate the model architecture for final_mlp_model_sep
                final_mlp_model_sep = mb.add_proj_head(final_mlp_model_sep_stage1, output_dim=1, freeze_features=freeze,
                                                       pds=pds)

                final_mlp_model_sep.compile(
                    optimizer=AdamW(learning_rate=learning_rate,
                                    weight_decay=weight_decay,
                                    beta_1=momentum_beta1),
                    loss={'forecast_head': get_loss(loss_key)})  # Compile the model just like before
                # Train on the full dataset
                final_mlp_model_sep.fit(X_train, {'forecast_head': y_train},
                                        epochs=optimal_epochs,
                                        batch_size=batch_size,
                                        verbose=1)

                # evaluate the model on test cme_files
                error_mae = evaluate_model(final_mlp_model_sep, X_test, y_test)
                print(f'mae error: {error_mae}')
                # Log the MAE error to wandb
                wandb.log({"mae_error": error_mae})

                # Log t-SNE plot for training
                # Log the training t-SNE plot to wandb
                stage2_file_path = plot_tsne_extended(final_mlp_model_sep,
                                                      X_train,
                                                      y_train,
                                                      title, 'stage2_training',
                                                      save_tag=current_time)
                wandb.log({'stage2_tsne_training_plot': wandb.Image(stage2_file_path)})
                print('stage2_file_path: ' + stage2_file_path)

                # Log t-SNE plot for testing
                # Log the testing t-SNE plot to wandb
                stage2_file_path = plot_tsne_extended(final_mlp_model_sep,
                                                      X_test,
                                                      y_test,
                                                      title, 'stage2_testing',
                                                      save_tag=current_time)
                wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage2_file_path)})
                print('stage2_file_path: ' + stage2_file_path)

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
                    wandb.log({f'testing_{filename}': wandb.Image(filename)})

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
