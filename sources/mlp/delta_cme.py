import os
from datetime import datetime

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
import numpy as np

from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    evaluate_model,
    evaluate_model_cond,
    process_sep_events,
    get_loss,
    reshape_X, filter_ds, stratified_split, plot_error_hist)


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for cme_speed_threshold in [0]:
            for alpha in np.arange(0.3, 0.6, 1):
                for add_slope in [False]:
                    # PARAMS
                    # inputs_to_use = ['e0.5']
                    # add_slope = True
                    outputs_to_use = ['delta_p']

                    # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                    inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                    # Construct the title
                    title = f'MLP_{inputs_str}_slope{str(add_slope)}_alpha{alpha:.2f}_CME{cme_speed_threshold}'

                    # Replace any other characters that are not suitable for filenames (if any)
                    title = title.replace(' ', '_').replace(':', '_')

                    # Create a unique experiment name with a timestamp
                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                    experiment_name = f'{title}_{current_time}'

                    # Set the early stopping patience and learning rate as variables
                    seed = 456789
                    tf.random.set_seed(seed)
                    np.random.seed(seed)
                    patience = 10000  # higher patience
                    learning_rate = 1e-2  # og learning rate

                    reduce_lr_on_plateau = ReduceLROnPlateau(
                        monitor='loss',
                        factor=0.9,
                        patience=1000,
                        verbose=1,
                        min_delta=1e-5,
                        min_lr=1e-4)

                    weight_decay = 1e-2  # higher weight decay
                    momentum_beta1 = 0.9  # higher momentum beta1
                    batch_size = 4096
                    epochs = 50000  # higher epochs
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
                    loss_key = 'mse'
                    target_change = ('delta_p' in outputs_to_use)
                    # print_batch_mse_cb = PrintBatchMSE()
                    rebalacing = True
                    alpha_rw = alpha
                    bandwidth = 4.42e-2
                    repr_dim = 128
                    output_dim = len(outputs_to_use)
                    dropout = 0.5
                    activation = None
                    norm = 'batch_norm'
                    cme_speed_threshold = cme_speed_threshold
                    residual = True
                    skipped_layers = 2
                    N = 500
                    lower_threshold = -0.5  # lower threshold for the delta_p
                    upper_threshold = 0.5  # upper threshold for the delta_p

                    # Initialize wandb
                    wandb.init(project="nasa-ts-delta-v6", name=experiment_name, config={
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
                        'architecture': 'mlp',
                        'cme_speed_threshold': cme_speed_threshold,
                        'residual': residual,
                        'skipped_layers': skipped_layers,
                        'ds_version': 6,
                    })

                    # set the root directory
                    root_dir = 'data/electron_cme_data_split_v5'
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
                        N=N, seed=seed)

                    X_test_filtered, y_test_filtered = filter_ds(
                        X_test, y_test,
                        low_threshold=lower_threshold,
                        high_threshold=upper_threshold,
                        N=N, seed=seed)

                    X_subtrain, y_subtrain, X_val, y_val = stratified_split(
                        X_train,
                        y_train,
                        shuffle=True,
                        seed=seed,
                        split=0.25,
                        debug=False)

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
                    delta_val = y_val[:, 0]
                    print(f'delta_train.shape: {delta_train.shape}')
                    print(f'delta_subtrain.shape: {delta_subtrain.shape}')
                    print(f'delta_val.shape: {delta_val.shape}')

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

                    print(f'rebalancing the validation set...')
                    min_norm_weight = 0.01 / len(delta_val)
                    y_val_weights = exDenseReweights(
                        X_val, delta_val,
                        alpha=alpha_rw, bw=bandwidth,
                        min_norm_weight=min_norm_weight,
                        debug=False).reweights
                    print(f'validation set rebalanced.')

                    # get the number of features
                    n_features = X_train.shape[1]
                    print(f'n_features: {n_features}')

                    # create the model
                    model_sep = create_mlp(
                        input_dim=n_features,
                        hiddens=hiddens,
                        repr_dim=repr_dim,
                        output_dim=output_dim,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        skipped_layers=skipped_layers
                    )
                    model_sep.summary()

                    print('Reshaping input for model')
                    X_subtrain = reshape_X(
                        X_subtrain,
                        [n_features],
                        inputs_to_use,
                        add_slope,
                        model_sep.name)

                    X_val = reshape_X(
                        X_val,
                        [n_features],
                        inputs_to_use,
                        add_slope,
                        model_sep.name)

                    X_train = reshape_X(
                        X_train,
                        [n_features],
                        inputs_to_use,
                        add_slope,
                        model_sep.name)

                    X_test = reshape_X(
                        X_test,
                        [n_features],
                        inputs_to_use,
                        add_slope,
                        model_sep.name)

                    # Define the EarlyStopping callback
                    early_stopping = EarlyStopping(
                        monitor='val_loss',
                        patience=patience,
                        verbose=1,
                        restore_best_weights=True, )

                    # Compile the model with the specified learning rate
                    model_sep.compile(
                        optimizer=AdamW(learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        beta_1=momentum_beta1),
                        loss={'forecast_head': get_loss(loss_key)}
                    )

                    # Train the model with the callback
                    history = model_sep.fit(
                        X_subtrain,
                        {'forecast_head': y_subtrain},
                        sample_weight=y_subtrain_weights,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, {'forecast_head': y_val}, y_val_weights),
                        callbacks=[
                            early_stopping,
                            reduce_lr_on_plateau,
                            WandbCallback(save_model=False)
                        ],
                        verbose=1
                    )

                    # Determine the optimal number of epochs from early stopping
                    optimal_epochs = early_stopping.stopped_epoch + 1  # Adjust for the offset

                    final_model_sep = create_mlp(
                        input_dim=n_features,
                        hiddens=hiddens,
                        repr_dim=repr_dim,
                        output_dim=output_dim,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        skipped_layers=skipped_layers
                    )

                    # Recreate the model architecture
                    final_model_sep.compile(
                        optimizer=AdamW(learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        beta_1=momentum_beta1),
                        loss={'forecast_head': get_loss(loss_key)}
                    )

                    # Train on the full dataset
                    final_model_sep.fit(
                        X_train,
                        {'forecast_head': y_train},
                        sample_weight=y_train_weights,
                        epochs=optimal_epochs,
                        batch_size=batch_size,
                        callbacks=[
                            reduce_lr_on_plateau,
                            WandbCallback(save_model=False)
                        ],
                        verbose=1
                    )

                    # evaluate the model on test cme_files
                    error_mae = evaluate_model(final_model_sep, X_test, y_test)
                    print(f'mae error: {error_mae}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error": error_mae})

                    # evaluate the model on training cme_files
                    error_mae_train = evaluate_model(final_model_sep, X_train, y_train)
                    print(f'mae error train: {error_mae_train}')
                    # Log the MAE error to wandb
                    wandb.log({"train_mae_error": error_mae_train})

                    # Process SEP event files in the specified directory
                    test_directory = root_dir + '/testing'
                    filenames = process_sep_events(
                        test_directory,
                        final_model_sep,
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
                        final_model_sep,
                        title=title,
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        show_avsp=True,
                        prefix='training',
                        using_cme=True,
                        cme_speed_threshold=cme_speed_threshold)

                    # Log the plot to wandb
                    for filename in filenames:
                        log_title = os.path.basename(filename)
                        wandb.log({f'training_{log_title}': wandb.Image(filename)})

                    # evaluate the model on test cme_files
                    above_threshold = 0.1
                    error_mae_cond = evaluate_model_cond(
                        final_model_sep, X_test, y_test, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 test: {error_mae_cond}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error_cond_test": error_mae_cond})

                    # evaluate the model on training cme_files
                    error_mae_cond_train = evaluate_model_cond(
                        final_model_sep, X_train, y_train, above_threshold=above_threshold)

                    print(f'mae error delta >= 0.1 train: {error_mae_cond_train}')
                    # Log the MAE error to wandb
                    wandb.log({"mae_error_cond_train": error_mae_cond_train})
                    # Log the MAE error to wandb

                    # Evaluate the model correlation with colored
                    file_path = plot_repr_corr_dist(
                        final_model_sep,
                        X_train_filtered, y_train_filtered,
                        title + "_training",
                        model_type='features_reg'
                    )
                    wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    file_path = plot_repr_corr_dist(
                        final_model_sep,
                        X_test_filtered, y_test_filtered,
                        title + "_test",
                        model_type='features_reg'
                    )
                    wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    # Log t-SNE plot
                    # Log the training t-SNE plot to wandb
                    stage1_file_path = plot_tsne_delta(
                        final_model_sep,
                        X_train_filtered, y_train_filtered, title,
                        'stage2_training',
                        model_type='features_reg',
                        save_tag=current_time, seed=seed)
                    wandb.log({'stage2_tsne_training_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Log the testing t-SNE plot to wandb
                    stage1_file_path = plot_tsne_delta(
                        final_model_sep,
                        X_test_filtered, y_test_filtered, title,
                        'stage2_testing',
                        model_type='features_reg',
                        save_tag=current_time, seed=seed)
                    wandb.log({'stage2_tsne_testing_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Plot the error histograms
                    filename = plot_error_hist(
                        final_model_sep,
                        X_train, y_train,
                        sample_weights=None,
                        title=title,
                        prefix='training')
                    wandb.log({"training_error_hist": wandb.Image(filename)})

                    # Plot the error weighted histograms
                    filename = plot_error_hist(
                        final_model_sep,
                        X_train, y_train,
                        sample_weights=y_train_weights,
                        title=title,
                        prefix='training_weighted')
                    wandb.log({"training_weighted_error_hist": wandb.Image(filename)})

                    # Plot the error histograms on the testing set
                    filename = plot_error_hist(
                        final_model_sep,
                        X_test, y_test,
                        sample_weights=None,
                        title=title,
                        prefix='testing')
                    wandb.log({"testing_error_hist": wandb.Image(filename)})

                    # Finish the wandb run
                    wandb.finish()


if __name__ == '__main__':
    main()
