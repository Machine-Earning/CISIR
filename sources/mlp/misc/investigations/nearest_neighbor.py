import os
from datetime import datetime
from typing import List, Tuple

import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow_addons.optimizers import AdamW
from wandb.keras import WandbCallback
import numpy as np
from sklearn.neighbors import NearestNeighbors

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
from modules.training.DenseReweights import exDenseReweights
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp_moe,
    evaluate_model,
    evaluate_model_cond,
    process_sep_events,
    get_loss,
    filter_ds, stratified_split, plot_error_hist)

from modules.shared.globals import *

def find_k_nearest_neighbors(
    X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, predictions: np.ndarray, k: int
) -> None:
    """
    Find the k nearest neighbors for each point in the test set with target labels greater than 0.5.

    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        y_test (np.ndarray): Test data target labels.
        predictions (np.ndarray): Model predictions for the test data.
        k (int): Number of nearest neighbors to find.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)

    large_positives_indices = np.where(y_test[:, 0] > 0.5)[0]
    for idx in large_positives_indices:
        distances, indices = nbrs.kneighbors([X_test[idx]])
        print(f"\nTest point {idx} (target label: {y_test[idx][0]}, predicted label: {predictions[idx][0]})")
        for i, (dist, neighbor_idx) in enumerate(zip(distances[0], indices[0])):
            print(f"Neighbor {i+1}: Distance={dist:.4f}, Target Label={y_test[neighbor_idx][0]}, Predicted Label={predictions[neighbor_idx][0]}")


def main() -> None:
    """
    Main function to run the E-MLP model.
    """
    for seed in SEEDS:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha in [0.5]:
                    for add_slope in ADD_SLOPE:
                        # PARAMS
                        outputs_to_use = OUTPUTS_TO_USE

                        inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                        title = f'Invesf_MLP__{inputs_str}_slope{str(add_slope)}_alpha{alpha:.2f}_CME{cme_speed_threshold}'
                        title = title.replace(' ', '_').replace(':', '_')
                        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                        experiment_name = f'{title}_{current_time}'

                        tf.random.set_seed(seed)
                        np.random.seed(seed)
                        patience = PATIENCE
                        learning_rate = START_LR_FT

                        reduce_lr_on_plateau = ReduceLROnPlateau(
                            monitor=LR_CB_MONITOR,
                            factor=LR_CB_FACTOR,
                            patience=LR_CB_PATIENCE,
                            verbose=VERBOSE,
                            min_delta=LR_CB_MIN_DELTA,
                            min_lr=LR_CB_MIN_LR)

                        weight_decay = WEIGHT_DECAY
                        momentum_beta1 = MOMENTUM_BETA1
                        batch_size = BATCH_SIZE
                        epochs = EPOCHS
                        hiddens = MLP_HIDDENS
                        hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                        loss_key = LOSS_KEY
                        target_change = ('delta_p' in outputs_to_use)
                        alpha_rw = alpha
                        bandwidth = BANDWIDTH
                        repr_dim = REPR_DIM
                        output_dim = len(outputs_to_use)
                        dropout = DROPOUT
                        activation = ACTIVATION
                        norm = NORM
                        cme_speed_threshold = cme_speed_threshold
                        residual = RESIDUAL
                        skipped_layers = SKIPPED_LAYERS
                        N = N_FILTERED
                        lower_threshold = LOWER_THRESHOLD
                        upper_threshold = UPPER_THRESHOLD
                        mae_plus_threshold = MAE_PLUS_THRESHOLD

                        wandb.init(project="nasa-ts-delta-v6", name=experiment_name, config={
                            "inputs_to_use": inputs_to_use,
                            "add_slope": add_slope,
                            "patience": patience,
                            "learning_rate": learning_rate,
                            "weight_decay": weight_decay,
                            "momentum_beta1": momentum_beta1,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "hiddens": hiddens_str,
                            "loss": loss_key,
                            "target_change": target_change,
                            "seed": seed,
                            "alpha_rw": alpha_rw,
                            "bandwidth": bandwidth,
                            "reciprocal_reweight": RECIPROCAL_WEIGHTS,
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
                            'ds_version': DS_VERSION,
                            'mae_plus_th': mae_plus_threshold
                        })

                        root_dir = DS_PATH
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

                        print(f'X_train.shape: {X_train.shape}')
                        print(f'y_train.shape: {y_train.shape}')
                        print(f'X_test.shape: {X_test.shape}')
                        print(f'y_test.shape: {y_test.shape}')

                        delta_train = y_train[:, 0]
                        delta_test = y_test[:, 0]
                        print(f'delta_train.shape: {delta_train.shape}')
                        print(f'delta_test.shape: {delta_test.shape}')

                        print(f'rebalancing the training set...')
                        min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
                        y_train_weights = exDenseReweights(
                            X_train, delta_train,
                            alpha=alpha_rw, bw=bandwidth,
                            min_norm_weight=min_norm_weight,
                            debug=False).reweights
                        print(f'training set rebalanced.')

                        print(f'rebalancing the validation set...')
                        min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                        y_test_weights = exDenseReweights(
                            X_test, delta_test,
                            alpha=COMMON_VAL_ALPHA, bw=bandwidth,
                            min_norm_weight=min_norm_weight,
                            debug=False).reweights
                        print(f'validation set rebalanced.')

                        n_features = X_train.shape[1]
                        print(f'n_features: {n_features}')

                        expert_high_path = '/home1/jmoukpe2016/keras-functional-api/inves_model_weights_Inves_MLP__e0_5_e1_8_p_slopeFalse_alpha1.00_CME0_20240701-235026_reg_moe.h5'
                        expert_low_path = '/home1/jmoukpe2016/keras-functional-api/inves_model_weights_Inves_MLP__e0_5_e1_8_p_slopeFalse_alpha0.00_CME0_20240701-235026_reg_moe.h5'
                        router_hiddens = MLP_HIDDENS
                        temperature = 0.1

                        model_sep = create_mlp_moe(
                            input_dim=n_features,
                            hiddens=hiddens,
                            repr_dim=repr_dim,
                            output_dim=output_dim,
                            dropout_rate=dropout,
                            activation=activation,
                            norm=norm,
                            residual=residual,
                            skipped_layers=skipped_layers,
                            expert_high_path=expert_high_path,
                            expert_low_path=expert_low_path,
                            router_hiddens=router_hiddens,
                            freeze_experts=True,
                            temperature=temperature
                        )
                        model_sep.summary()

                        early_stopping = EarlyStopping(
                            monitor=ES_CB_MONITOR,
                            patience=patience,
                            verbose=VERBOSE,
                            restore_best_weights=True)

                        best_weights_filepath = f"inves_model_weights_{experiment_name}_reg_moe.h5"
                        model_checkpoint = ModelCheckpoint(
                            filepath=best_weights_filepath,
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='min',
                            save_best_only=True,
                            verbose=1
                        )

                        model_sep.compile(
                            optimizer=AdamW(
                                learning_rate=learning_rate,
                                weight_decay=weight_decay,
                                beta_1=momentum_beta1
                            ),
                            loss={'forecast_head': get_loss(loss_key)}
                        )

                        model_sep.fit(
                            X_train,
                            {'forecast_head': y_train},
                            sample_weight=y_train_weights,
                            epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, {'forecast_head': y_test}, y_test_weights),
                            callbacks=[
                                early_stopping,
                                reduce_lr_on_plateau,
                                model_checkpoint,
                                WandbCallback(save_model=WANDB_SAVE_MODEL)
                            ],
                            verbose=VERBOSE
                        )

                        model_sep.load_weights(best_weights_filepath)
                        print(f"Model weights are saved in {best_weights_filepath}")

                        # Evaluate the model and get predictions
                        error_mae = evaluate_model(model_sep, X_test, y_test)
                        print(f'mae error: {error_mae}')
                        wandb.log({"mae_error": error_mae})

                        error_mae_train = evaluate_model(model_sep, X_train, y_train)
                        print(f'mae error train: {error_mae_train}')
                        wandb.log({"train_mae_error": error_mae_train})

                        # Get model predictions
                        predictions = model_sep.predict(X_test)['forecast_head']

                        # Find and print the k nearest neighbors for large positives in the test set
                        k = 5  # Set the number of nearest neighbors to find
                        find_k_nearest_neighbors(X_train, X_test, y_test, predictions, k)

                        test_directory = root_dir + '/testing'
                        filenames = process_sep_events(
                            test_directory,
                            model_sep,
                            title=title,
                            inputs_to_use=inputs_to_use,
                            add_slope=add_slope,
                            outputs_to_use=outputs_to_use,
                            show_avsp=True,
                            using_cme=True,
                            cme_speed_threshold=cme_speed_threshold)

                        for filename in filenames:
                            log_title = os.path.basename(filename)
                            wandb.log({f'testing_{log_title}': wandb.Image(filename)})

                        test_directory = root_dir + '/training'
                        filenames = process_sep_events(
                            test_directory,
                            model_sep,
                            title=title,
                            inputs_to_use=inputs_to_use,
                            add_slope=add_slope,
                            outputs_to_use=outputs_to_use,
                            show_avsp=True,
                            prefix='training',
                            using_cme=True,
                            cme_speed_threshold=cme_speed_threshold)

                        for filename in filenames:
                            log_title = os.path.basename(filename)
                            wandb.log({f'training_{log_title}': wandb.Image(filename)})

                        above_threshold = mae_plus_threshold
                        error_mae_cond = evaluate_model_cond(
                            model_sep, X_test, y_test, above_threshold=above_threshold)

                        print(f'mae error delta >= 0.1 test: {error_mae_cond}')
                        wandb.log({"mae_error_cond_test": error_mae_cond})

                        error_mae_cond_train = evaluate_model_cond(
                            model_sep, X_train, y_train, above_threshold=above_threshold)

                        print(f'mae error delta >= 0.1 train: {error_mae_cond_train}')
                        wandb.log({"mae_error_cond_train": error_mae_cond_train})

                        wandb.finish()


if __name__ == '__main__':
    main()
