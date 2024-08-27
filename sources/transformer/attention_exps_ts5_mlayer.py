import os
from datetime import datetime

import numpy as np
import pandas as pd
import wandb
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow_addons.optimizers import AdamW
from wandb.integration.keras import WandbCallback

from modules.reweighting.exDenseReweightsD import exDenseReweightsD
from modules.shared.globals import *
from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    evaluate_mae, process_sep_events, evaluate_pcc, build_dataset, mse_pcc)
from modules.training.ts_modeling import set_seed
# Importing the Blocks
from sources.transformer.modules import *

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

devices = tf.config.list_physical_devices('GPU')
print(f'devices: {devices}')
set_seed(SEEDS[0])  # Set seed for reproducibility

# Hyperparameters
root_dir = DS_PATH
inputs_to_use = INPUTS_TO_USE[0]
outputs_to_use = OUTPUTS_TO_USE
cme_speed_threshold = -1  # CME_SPEED_THRESHOLD[0]
using_cme = True if cme_speed_threshold >= 0 else False
add_slope = False
hiddens = [128 for _ in range(7)]
# blocks = [128, 64, 128]
blocks = []

a = 1
LR = 3e-3
EPOCHS = int(1e4)
BS = 4096
PATIENCE = int(3e3)
bandwidth = BANDWIDTH
residual = True
skipped_layers = 2
weight_decay = 1e-8
lambda_ = 0.05


def create_model(input_shape: int, rho: float, hiddens: list, blocks: list) -> Model:
    """
    Create a model using the specified block class.

    Args:
        input_shape (int): The shape of the input tensor.
        rho (float): SAM regularization parameter. If rho > 0, a SAM model is created.
        hiddens (list): List of hidden units for each block.
        blocks (list): List specifying the number of blocks and their configurations.

    Returns:
        Model: The Keras model.
    """
    inputs = Input(shape=(input_shape,))
    x = inputs

    # Iterate through the number of blockshyper
    for i in range(len(blocks)):
        block = TanhAttentiveBlock(
            attn_hidden_units=hiddens,
            attn_hidden_activation='leaky_relu',
            output_activation='linear',
            attn_skipped_layers=skipped_layers,
            attn_residual=residual,
            attn_norm=None,
            output_dim=blocks[i],
            a=a)
        x = block(x)['output']  # Apply the block to the input

    # The final output is the output of the last block
    output_block = TanhAttentiveBlock(
        attn_hidden_units=hiddens,
        attn_hidden_activation='leaky_relu',
        output_activation='linear',
        attn_skipped_layers=skipped_layers,
        attn_residual=residual,
        attn_norm=None,
        output_dim=1,
        a=a)
    # final output
    outputs = output_block(x)  # dict of outputs and attention scores

    # If rho is greater than 0, wrap the model in SAMModel
    if rho > 0:
        model = SAMModel(inputs=inputs, outputs=outputs, rho=rho)
    else:
        model = Model(inputs=inputs, outputs=outputs)

    return model


def train_and_print_results(
        block_name: str,
        model: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        x_debug: np.ndarray = None,
        y_debug: np.ndarray = None,
        learning_rate: float = 0.003,
        weight_decay: float = 0,
        epochs: int = 500,
        batch_size: int = 32,
        patience: int = 100,
        debug: bool = True,
        time: str = "",
        train_weights_dict: dict = None,
        val_weights_dict: dict = None
) -> None:
    """
    Train the model and print the results, including learned weights and attention scores.

    Args:
        block_name (str): Name of the attention block.
        model (Model): The Keras model to train.
        x_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        y_train_weights (np.ndarray): Training sample weights
        x_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.
        y_test_weights (np.ndarray): Test sample weights
        x_debug (np.ndarray): Debug features.
        y_debug (np.ndarray): Debug labels.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        patience (int): Patience for early stopping.
        time (str): The current time for the experiment.
        train_weights_dict: Dictionary containing the weights for the training set
        val_weights_dict: Dictionary containing the weights for the validation set
    """
    model.compile(
        optimizer=AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss={'output': lambda y_true, y_pred: mse_pcc(
            y_true, y_pred,
            lambda_factor=lambda_,
            train_weight_dict=train_weights_dict,
            val_weight_dict=val_weights_dict
        )},
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=VERBOSE
    )

    model.fit(
        x_train,
        {'output': y_train},
        epochs=epochs,
        batch_size=batch_size,
        verbose=VERBOSE,
        validation_data=(x_test, {'output': y_test}),
        callbacks=[early_stopping, WandbCallback(save_model=False)],
    )

    # Evaluate the model on the debug set
    debug_mae = evaluate_mae(model, x_debug, y_debug, use_dict=True)
    print(f"MAE debug: {debug_mae}")
    # log the results
    wandb.log({"debug_mae": debug_mae})

    if debug:
        # Predict on initial data
        predictions = model.predict(x_debug)
        output_predictions = predictions['output']
        attention_scores = predictions['attention_scores']
        print("Predictions on initial data:")

        # if block_name != '2':
        #     # Retrieve the weights and bias of the last dense layer
        #     dense_layer_weights, dense_layer_bias = model.layers[-1].get_dense_weights()
        #     print('weights')
        #     print(dense_layer_weights)
        #     print(dense_layer_bias)
        # else:
        # dense_layer_weights, dense_layer_bias = np.array([[1], [1]]), np.array([0])

        results = []
        for pred, true, inp, attn in zip(output_predictions, y_debug, x_debug, attention_scores):
            # attention_weighted_values = [a * w for a, w in zip(attn, 
            #                                                 #    dense_layer_weights[:, 0]
            #                                                    )]
            results.append(
                list(inp) + [true[0], pred[0]]
                + attn.tolist()
                # + attention_weighted_values
                # + [dense_layer_bias[0]]
                # + dense_layer_weights[:, 0].tolist()
            )

        # Print results in a table
        headers = (
                [f'x{i + 1}' for i in range(len(inp))]
                + ['True y', 'Predicted y']
                + [f'Attention_{i + 1}' for i in range(attention_scores.shape[1])]
            # + [f'Attention_Weight_{i + 1}' for i in range(attention_scores.shape[1])]
            # + ['Bias'] + [f'Weight_{i + 1}' for i in range(dense_layer_weights.shape[0])]
        )

        df_results = pd.DataFrame(results, columns=headers)
        print(df_results)
        # wandb.log({f"results_{block_name}_{time}": df_results})  # so cool you can log dataframes
        # Save the results DataFrame to a CSV file
        df_results.to_csv(f"results_{block_name}_{time}.csv", index=False)
        print(f"Results saved to results_{block_name}_{time}.csv")

        # evaluate the model on test cme_files
        error_mae = evaluate_mae(model, x_test, y_test, use_dict=True)
        print(f'mae error: {error_mae}')
        wandb.log({"mae": error_mae})

        # evaluate the model on training cme_files
        error_mae_train = evaluate_mae(model, x_train, y_train, use_dict=True)
        print(f'mae error train: {error_mae_train}')
        wandb.log({"train_mae": error_mae_train})

        # evaluate the model correlation on test set
        error_pcc = evaluate_pcc(model, x_test, y_test, use_dict=True)
        print(f'pcc error: {error_pcc}')
        wandb.log({"pcc": error_pcc})

        # evaluate the model correlation on training set
        error_pcc_train = evaluate_pcc(model, x_train, y_train, use_dict=True)
        print(f'pcc error train: {error_pcc_train}')
        wandb.log({"train_pcc": error_pcc_train})

        error_pcc_debug = evaluate_pcc(model, x_debug, y_debug, use_dict=True)
        print(f'pcc error debug: {error_pcc_debug}')
        wandb.log({"debug_pcc": error_pcc_debug})

        # evaluate the model on test cme_files
        above_threshold = 0.5
        error_mae_cond = evaluate_mae(
            model, x_test, y_test, above_threshold=above_threshold, use_dict=True)
        print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')
        wandb.log({"mae+": error_mae_cond})

        error_mae_train_cond = evaluate_mae(
            model, x_train, y_train, above_threshold=above_threshold, use_dict=True)
        print(f'mae error delta >= {above_threshold} train: {error_mae_train_cond}')
        wandb.log({"train_mae+": error_mae_train_cond})

        # evaluate the model correlation for rare samples on test set
        error_pcc_cond = evaluate_pcc(
            model, x_test, y_test, above_threshold=above_threshold, use_dict=True)
        print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
        wandb.log({"pcc+": error_pcc_cond})

        # evaluate the model correlation for rare samples on training set
        error_pcc_cond_train = evaluate_pcc(
            model, x_train, y_train, above_threshold=above_threshold, use_dict=True)
        print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
        wandb.log({"train_pcc+": error_pcc_cond_train})

        # debug
        error_mae_cond_debug = evaluate_mae(
            model, x_debug, y_debug, above_threshold=above_threshold, use_dict=True)
        print(f'mae error delta >= {above_threshold} test: {error_mae_cond_debug}')
        wandb.log({"debug_mae+": error_mae_cond_debug})

        # evaluate the model correlation for rare samples on debug set
        error_pcc_cond_debug = evaluate_pcc(
            model, x_debug, y_debug, above_threshold=above_threshold, use_dict=True)
        print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond_debug}')
        wandb.log({"debug_pcc+": error_pcc_cond_debug})

        # Process SEP event files in the specified directory
        test_directory = root_dir + '/testing'
        filenames = process_sep_events(
            test_directory,
            model,
            title=block_name,
            inputs_to_use=inputs_to_use,
            add_slope=add_slope,
            outputs_to_use=outputs_to_use,
            show_avsp=True,
            using_cme=using_cme,
            cme_speed_threshold=cme_speed_threshold,
            use_dict=True)

        # Log the plot to wandb
        for filename in filenames:
            log_title = os.path.basename(filename)
            wandb.log({f'testing_{log_title}': wandb.Image(filename)})

        # Process SEP event files in the specified directory
        test_directory = root_dir + '/training'
        filenames = process_sep_events(
            test_directory,
            model,
            title=block_name,
            inputs_to_use=inputs_to_use,
            add_slope=add_slope,
            outputs_to_use=outputs_to_use,
            show_avsp=True,
            prefix='training',
            using_cme=using_cme,
            cme_speed_threshold=cme_speed_threshold,
            use_dict=True)

        # Log the plot to wandb
        for filename in filenames:
            log_title = os.path.basename(filename)
            wandb.log({f'training_{log_title}': wandb.Image(filename)})


for alpha_val in [0.75]:
    for alpha in [0.25]:
        for rho in [0]:
            # for alpha in np.arange(1.1, 1.5, 0.1):
            # for alpha in np.arange(0, 1.1, 0.1):
            # Create a unique experiment name with a timestamp
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f'Attention_Type7_{current_time}_alphaVal{alpha_val}_alpha{alpha}_3L'

            # dataset
            # build the dataset
            x_train, y_train = build_dataset(
                root_dir + '/training',
                inputs_to_use=inputs_to_use,
                add_slope=add_slope,
                outputs_to_use=outputs_to_use,
                cme_speed_threshold=cme_speed_threshold,
                shuffle_data=True)

            x_test, y_test = build_dataset(
                root_dir + '/testing',
                inputs_to_use=inputs_to_use,
                add_slope=add_slope,
                outputs_to_use=outputs_to_use,
                cme_speed_threshold=cme_speed_threshold)

            x_debug, y_debug = build_dataset(
                root_dir + '/debug',
                inputs_to_use=inputs_to_use,
                add_slope=add_slope,
                outputs_to_use=outputs_to_use,
                cme_speed_threshold=cme_speed_threshold)

            # Verify data shapes
            print(f"Shape of x_train: {x_train.shape}")
            print(f"Shape of y_train: {y_train.shape}")
            print(f"Shape of x_test: {x_test.shape}")
            print(f"Shape of y_test: {y_test.shape}")
            print(f"Shape of x_debug: {x_debug.shape}")
            print(f"Shape of y_debug: {y_debug.shape}")

            # Dense Loss
            # Compute the sample weights
            delta_train = y_train[:, 0]
            delta_test = y_test[:, 0]
            print(f'delta_train.shape: {delta_train.shape}')
            print(f'delta_test.shape: {delta_test.shape}')

            print(f'rebalancing the training set...')
            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
            train_weights_dict = exDenseReweightsD(
                x_train, delta_train,
                alpha=alpha, bw=bandwidth,
                min_norm_weight=min_norm_weight,
                debug=False).label_reweight_dict
            print(f'training set rebalanced.')

            print(f'rebalancing the test set...')
            min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
            test_weights_dict = exDenseReweightsD(
                x_test, delta_test,
                alpha=alpha_val, bw=bandwidth,
                min_norm_weight=min_norm_weight,
                debug=False).label_reweight_dict
            print(f'test set rebalanced.')

            # Training and printing results for each attention type
            input_dim = x_train.shape[1]
            block_classes = TanhAttentiveBlock

            # Initialize wandb
            wandb.init(project="attention-exps-5.6", name=experiment_name, config={
                "learning_rate": LR,
                "epochs": EPOCHS,
                "batch_size": BS,
                "attention_type": 7,
                "patience": PATIENCE,
                'attn_hiddens': (", ".join(map(str, hiddens))).replace(', ', '_'),
                'block_hiddens': (", ".join(map(str, blocks))).replace(', ', '_'),
                'slope': add_slope,
                'a': a,
                'alpha': alpha,
                'alpha_val': alpha_val,
                'bandwidth': bandwidth,
                'weight_dacay': weight_decay,
                'residual': residual,
                "skipped_layers": skipped_layers,
                'sam_rho': rho,
                'loss_key': 'mse_pcc',
            })

            print(f"\nAttention Type 7")
            # model = create_model(input_dim, rho=rho, hiddens=hiddens, blocks=blocks)
            model = create_attentive_model(
                input_dim=input_dim,
                output_dim=1,
                attn_hidden_units=hiddens,
                attn_hidden_activation='leaky_relu',
                hidden_blocks=blocks,
                skipped_blocks=skipped_layers,
                residual=residual,
                sam_rho=rho,
            )
            model.summary()

            # tf.keras.utils.plot_model(model, to_file=f'./model_{i}.png', show_shapes=True)
            train_and_print_results(
                "7", model,
                x_train, y_train,
                x_test, y_test,
                x_debug=x_debug, y_debug=y_debug,
                learning_rate=LR, epochs=EPOCHS,
                batch_size=BS, patience=PATIENCE,
                weight_decay=weight_decay,
                time=current_time,
                train_weights_dict=train_weights_dict,
                val_weights_dict=test_weights_dict
            )

            # Finish the wandb run
            wandb.finish()
