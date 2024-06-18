import os
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback

from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    create_1dcnn,
    create_hybrid_model,
    evaluate_model,
    evaluate_model_cond,
    process_sep_events,
    get_loss,
    reshape_X)


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for add_slope in [True, False]:
            for alpha in np.arange(0.1, 1.6, 0.25):
                for cme_speed_threshold in [0, 500]:

                    # PARAMS
                    # inputs_to_use = ['e0.5']
                    # add_slope = True
                    outputs_to_use = ['delta_p']

                    # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                    inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                    # Construct the title
                    title = f'1DCNN_{inputs_str}_slope{str(add_slope)}_alpha{alpha:.2f}_CME{cme_speed_threshold}'

                    # Replace any other characters that are not suitable for filenames (if any)
                    title = title.replace(' ', '_').replace(':', '_')

                    # Create a unique experiment name with a timestamp
                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                    experiment_name = f'{title}_{current_time}'

                    # Set the early stopping patience and learning rate as variables
                    seed = 456789
                    tf.random.set_seed(seed)
                    np.random.seed(seed)
                    patience = 5000  # higher patience
                    learning_rate = 5e-3  # og learning rate
                    # initial_learning_rate = 3e-3
                    # final_learning_rate = 3e-7
                    # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / 3000)
                    # steps_per_epoch = int(20000 / 8)

                    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                    #     initial_learning_rate=initial_learning_rate,
                    #     decay_steps=steps_per_epoch,
                    #     decay_rate=learning_rate_decay_factor,
                    #     staircase=True)

                    reduce_lr_on_plateau = ReduceLROnPlateau(
                        monitor='loss',
                        factor=0.5,
                        patience=300,
                        verbose=1,
                        min_delta=1e-5,
                        min_lr=1e-10)

                    weight_decay = 1e-6  # higher weight decay
                    momentum_beta1 = 0.9  # higher momentum beta1
                    batch_size = 4096
                    epochs = 50000  # higher epochs
                    hiddens = [
                        (32, 10, 1, 'none', 0),  # Conv1: Start with broad features
                        (64, 8, 1, 'max', 2),  # Conv2 + Pool: Start to reduce and capture features
                        (64, 7, 1, 'none', 0),  # Conv3: Further detail without reducing dimension
                        (128, 5, 1, 'none', 0),  # Conv4: Increase filters, capture more refined features
                        (128, 5, 1, 'max', 2),  # Conv5 + Pool: Reduce dimension and capture more refined features
                        (256, 3, 1, 'none', 0),  # Conv6: Increase depth without immediate pooling
                        (256, 3, 1, 'none', 0),  # Conv7: Continue with high capacity, no dimension reduction
                        (512, 3, 1, 'max', 2),  # Conv8 + Pool: Use max pooling for stronger feature selection
                        (512, 3, 1, 'none', 0),  # Conv9: Maintain capacity, focusing on detailed features
                        # Note: Assuming a reduction towards dense layers after Conv9
                    ]
                    hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                    loss_key = 'mse'
                    target_change = ('delta_p' in outputs_to_use)
                    # print_batch_mse_cb = PrintBatchMSE()
                    rebalacing = True
                    alpha_rw = alpha
                    bandwidth = 0.099
                    repr_dim = 9
                    output_dim = len(outputs_to_use)
                    dropout = 0.5
                    activation = None
                    norm = 'batch_norm'
                    cme_speed_threshold = cme_speed_threshold

                    mlp_hiddens = [128, 64, 32]
                    mlp_repr_dim = 9
                    final_hiddens = [36, 18]
                    final_repr_dim = 9

                    # Initialize wandb
                    wandb.init(project="nasa-ts-delta", name=experiment_name, config={
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
                        "target_change": target_change,
                        "printing_batch_mse": False,
                        "seed": seed,
                        "rebalancing": rebalacing,
                        "alpha_rw": alpha_rw,
                        "bandwidth": bandwidth,
                        "reciprocal_reweight": True,
                        "repr_dim": repr_dim,
                        "dropout": dropout,
                        "activation": 'LeakyReLU',
                        "norm": norm,
                        'optimizer': 'adamw',
                        'output_dim': output_dim,
                        'architecture': '1dcnn',
                        'cme_speed_threshold': cme_speed_threshold
                    })

                    # set the root directory
                    root_dir = 'data/electron_cme_data_split'
                    # build the dataset
                    X_train, y_train = build_dataset(root_dir + '/training',
                                                     inputs_to_use=inputs_to_use,
                                                     add_slope=add_slope,
                                                     outputs_to_use=outputs_to_use,
                                                     cme_speed_threshold=cme_speed_threshold)
                    X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining',
                                                           inputs_to_use=inputs_to_use,
                                                           add_slope=add_slope,
                                                           outputs_to_use=outputs_to_use,
                                                           cme_speed_threshold=cme_speed_threshold)
                    X_test, y_test = build_dataset(root_dir + '/testing',
                                                   inputs_to_use=inputs_to_use,
                                                   add_slope=add_slope,
                                                   outputs_to_use=outputs_to_use,
                                                   cme_speed_threshold=cme_speed_threshold)
                    X_val, y_val = build_dataset(root_dir + '/validation',
                                                 inputs_to_use=inputs_to_use,
                                                 add_slope=add_slope,
                                                 outputs_to_use=outputs_to_use,
                                                 cme_speed_threshold=cme_speed_threshold)

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
                    print(f'delta_train.shape: {delta_train.shape}')
                    print(f'delta_subtrain.shape: {delta_subtrain.shape}')

                    print(f'rebalancing the training set...')
                    min_norm_weight = 0.01 / len(delta_train)
                    y_train_weights = exDenseReweights(
                        X_train, delta_train,
                        alpha=alpha_rw, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).reweights
                    print(f'training set rebalanced.')

                    print(f'rebalancing the subtraining set...')
                    min_norm_weight = 0.01 / len(delta_subtrain)
                    y_subtrain_weights = exDenseReweights(
                        X_subtrain, delta_subtrain,
                        alpha=alpha_rw, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).reweights
                    print(f'subtraining set rebalanced.')

                    # print a sample of the training cme_files
                    # print(f'X_train[0]: {X_train[0]}')
                    # print(f'y_train[0]: {y_train[0]}')

                    # get the number of features
                    if add_slope:
                        # n_features = [25] * len(inputs_to_use) * 2
                        n_features = [25] * len(inputs_to_use) + [24] * len(inputs_to_use)
                    else:
                        n_features = [25] * len(inputs_to_use)
                    print(f'n_features: {n_features}')

                    # calculating number of cme features
                    n_cme_features = 20 + len(inputs_to_use)

                    # create the model extractor
                    extractor_model_sep = create_1dcnn(
                        input_dims=n_features,
                        hiddens=hiddens,
                        repr_dim=repr_dim,
                        output_dim=0,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm
                    )
                    extractor_model_sep.summary()
                    # creating the hybrid model
                    mlp_model_sep = create_hybrid_model(
                        tsf_extractor=extractor_model_sep,
                        mlp_input_dim=n_cme_features,
                        output_dim=output_dim,
                        mlp_hiddens=mlp_hiddens,
                        mlp_repr_dim=mlp_repr_dim,
                        final_hiddens=final_hiddens,
                        repr_dim=final_repr_dim,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm,
                        name='hybrid'
                    )
                    mlp_model_sep.summary()

                    print('Reshaping input for model')
                    X_subtrain = reshape_X(
                        X_subtrain,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        'hybrid')

                    X_val = reshape_X(
                        X_val,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        'hybrid')

                    X_train = reshape_X(
                        X_train,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        'hybrid')

                    X_test = reshape_X(
                        X_test,
                        n_features,
                        inputs_to_use,
                        add_slope,
                        'hybrid')

                    # Define the EarlyStopping callback
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        verbose=1,
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
                                                callbacks=[
                                                    early_stopping,
                                                    reduce_lr_on_plateau,
                                                    WandbCallback(save_model=False)
                                                ])

                    # Plot the training and validation loss
                    plt.figure(figsize=(12, 6))
                    plt.plot(history.history['loss'], label='Training Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.title('Training and Validation Loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('Loss')
                    plt.legend()
                    # save the plot
                    plt.savefig(f'cnn_loss_{title}.png')

                    # Determine the optimal number of epochs from early stopping
                    optimal_epochs = early_stopping.stopped_epoch  + 1  # Adjust for the offset

                    # create the model extractor
                    final_extractor_model_sep = create_1dcnn(
                        input_dims=n_features,
                        hiddens=hiddens,
                        repr_dim=repr_dim,
                        output_dim=0,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm
                    )

                    # creating the hybrid model
                    final_mlp_model_sep = create_hybrid_model(
                        tsf_extractor=final_extractor_model_sep,
                        mlp_input_dim=n_cme_features,
                        output_dim=output_dim,
                        mlp_hiddens=mlp_hiddens,
                        mlp_repr_dim=mlp_repr_dim,
                        final_hiddens=final_hiddens,
                        repr_dim=final_repr_dim,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm,
                        name='hybrid'
                    )

                    # Recreate the model architecture
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
                        callbacks=[reduce_lr_on_plateau, WandbCallback(save_model=False)],
                        verbose=1)

                    # evaluate the model on test cme_files
                    error_mae = evaluate_model(final_mlp_model_sep, X_test, y_test)
                    print(f'mae error: {error_mae}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error": error_mae})

                    # evaluate the model on training cme_files
                    error_mae_train = evaluate_model(final_mlp_model_sep, X_train, y_train)
                    print(f'mae error train: {error_mae_train}')
                    # Log the MAE error to wandb
                    wandb.log({"train_mae_error": error_mae_train})

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/testing'
                    filenames = process_sep_events(
                        test_directory,
                        final_mlp_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True,
                        using_cme=True,
                        cme_speed_threshold=cme_speed_threshold)

                    # Log the plot to wandb
                    for filename in filenames:
                        log_title = os.path.basename(filename)
                        wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/training'
                    filenames = process_sep_events(
                        test_directory,
                        final_mlp_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True,
                        using_cme=True,
                        prefix='training',
                        cme_speed_threshold=cme_speed_threshold)

                    # Log the plot to wandb
                    for filename in filenames:
                        log_title = os.path.basename(filename)
                        wandb.log({f'training_{log_title}': wandb.Image(filename)})

                    # evaluate the model on test cme_files
                    above_threshold = 0.1
                    error_mae_cond = evaluate_model_cond(
                        final_mlp_model_sep, X_test, y_test, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 test: {error_mae_cond}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error_cond_test": error_mae_cond})

                    # evaluate the model on training cme_files
                    error_mae_cond_train = evaluate_model_cond(
                        final_mlp_model_sep, X_train, y_train, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 train: {error_mae_cond_train}')
                    # Log the MAE error to wandb

                    # Finish the wandb run
                    wandb.finish()


if __name__ == '__main__':
    main()
