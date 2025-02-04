import os
from datetime import datetime

import numpy as np
import wandb

from modules.shared.globals import *
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    process_sep_events,
    get_subset_ds,
    create_mlp,
    create_mlp_moe,
    set_seed
)


def main():
    """
    Main function to evaluate the E-MLP model
    :return:
    """

    eval_threshold = -1
    # moe_model_path = MOE_V2_PCC_CE_S2_A04_INVESTIGATION
    combiner_path = COMBINER_V2_PCC_CE_S2_A04_INVESTIGATION
    pos_expert_path = POS_EXPERT_PATH
    neg_expert_path = NEG_EXPERT_PATH
    zero_expert_path = NZ_EXPERT_PATH

    for seed in SEEDS:
        for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in [(0.4, 0.4, 0.0, 0.0)]:
            for rho in RHO_MOE:  # SAM
                inputs_to_use = INPUTS_TO_USE[0]
                cme_speed_threshold = CME_SPEED_THRESHOLD[0]
                add_slope = ADD_SLOPE[0]
                # PARAMS
                outputs_to_use = OUTPUTS_TO_USE
                lambda_factor = LAMBDA_FACTOR_MOE
                inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                title = f'mlp2_amse{alpha_mse:.2f}_v2_moe_cheat_pcc_ce_load_and_eval_A'
                title = title.replace(' ', '_').replace(':', '_')
                current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                experiment_name = f'{title}_{current_time}'

                set_seed(seed)
                patience = PATIENCE_MOE
                learning_rate = START_LR_MOE
                asym_type = ASYM_TYPE_MOE
                weight_decay = WEIGHT_DECAY_MOE
                momentum_beta1 = MOMENTUM_BETA1
                batch_size = BATCH_SIZE_NEG
                epochs = EPOCHS
                hiddens = MLP_HIDDENS
                hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                bandwidth = BANDWIDTH
                embed_dim = EMBED_DIM
                output_dim = len(outputs_to_use)
                dropout = DROPOUT
                activation = ACTIVATION
                norm = NORM
                cme_speed_threshold = cme_speed_threshold
                skip_repr = SKIP_REPR
                skipped_layers = SKIPPED_LAYERS
                mae_plus_threshold = MAE_PLUS_THRESHOLD
                hiddens = MLP_HIDDENS

                expert_paths = {
                    'plus': pos_expert_path,
                    'zero': zero_expert_path,
                    'minus': neg_expert_path,
                    'combiner': combiner_path
                }

                # Initialize wandb
                wandb.init(project="Jan-Report", name=experiment_name, config={
                    "inputs_to_use": inputs_to_use,
                    "add_slope": add_slope,
                    "patience": patience,
                    "learning_rate": learning_rate,
                    'min_lr': LR_CB_MIN_LR_MOE,
                    "weight_decay": weight_decay,
                    "momentum_beta1": momentum_beta1,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "hiddens": hiddens_str,
                    "loss": 'cmse',
                    "lambda": lambda_factor,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "alphaV_mse": alphaV_mse,
                    "alpha_pcc": alpha_pcc,
                    "alphaV_pcc": alphaV_pcc,
                    "bandwidth": bandwidth,
                    "embed_dim": embed_dim,
                    "dropout": dropout,
                    "activation": 'LeakyReLU',
                    "norm": norm,
                    'optimizer': 'adamw',
                    'output_dim': output_dim,
                    'architecture': 'mlp_moe',
                    'cme_speed_threshold': cme_speed_threshold,
                    'skip_repr': skip_repr,
                    'ds_version': DS_VERSION,
                    'mae_plus_th': mae_plus_threshold,
                    'sam_rho': rho,
                    'expert+_path': pos_expert_path,
                    'expert0_path': zero_expert_path,
                    'expert-_path': neg_expert_path,
                    'combiner_path': combiner_path,
                    # 'moe_model_path': moe_model_path,
                    'asym_type': asym_type
                })

                # Build datasets
                root_dir = DS_PATH
                X_train, y_train, logI_train, logI_prev_train = build_dataset(
                    root_dir + '/training',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold,
                    shuffle_data=True)

                # Get subset of training data
                X_train_subset, y_train_subset, _, _ = get_subset_ds(
                    X_train, y_train, eval_threshold, None, logI_train, logI_prev_train)

                # print the subset training set shapes
                print(f'X_train_subset.shape: {X_train_subset.shape}, y_train_subset.shape: {y_train_subset.shape}')
                # print the training set shapes
                print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}')

                # get the number of input features
                n_features = X_train.shape[1]
                print(f'n_features: {n_features}')

                X_test, y_test, logI_test, logI_prev_test = build_dataset(
                    root_dir + '/testing',
                    inputs_to_use=inputs_to_use,
                    add_slope=add_slope,
                    outputs_to_use=outputs_to_use,
                    cme_speed_threshold=cme_speed_threshold)

                # Get subset of test data with delta <= -0.4 for testing
                X_test_subset, y_test_subset, _, _ = get_subset_ds(
                    X_test, y_test, eval_threshold, None, logI_test, logI_prev_test)

                # print the subset test set shapes
                print(f'X_test_subset.shape: {X_test_subset.shape}, y_test_subset.shape: {y_test_subset.shape}')
                # print the test set shapes
                print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}')

                # Create experts using the create_mlp function - each expert outputs a single regression value
                expert_plus = create_mlp(
                    input_dim=n_features,
                    output_dim=1,
                    hiddens=hiddens,
                    skipped_layers=skipped_layers,
                    embed_dim=embed_dim,
                    skip_repr=skip_repr,
                    pretraining=False,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    name='expert_p'
                )
                expert_plus.load_weights(pos_expert_path)
                expert_plus.summary()

                expert_zero = create_mlp(
                    input_dim=n_features,
                    output_dim=1,
                    hiddens=hiddens,
                    skipped_layers=skipped_layers,
                    embed_dim=embed_dim,
                    skip_repr=skip_repr,
                    pretraining=False,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    name='expert_nz'
                )
                expert_zero.load_weights(zero_expert_path)
                expert_zero.summary()

                expert_minus = create_mlp(
                    input_dim=n_features,
                    output_dim=1,
                    hiddens=hiddens,
                    skipped_layers=skipped_layers,
                    embed_dim=embed_dim,
                    skip_repr=skip_repr,
                    pretraining=False,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    name='expert_m'
                )
                expert_minus.load_weights(neg_expert_path)
                expert_minus.summary()

                # Create combiner network - outputs class probabilities
                combiner = create_mlp(
                    input_dim=n_features,
                    output_dim=3,  # 3 classes: plus, mid, minus
                    hiddens=hiddens,
                    skipped_layers=skipped_layers,
                    embed_dim=embed_dim,
                    skip_repr=skip_repr,
                    pretraining=False,
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                    output_activation='norm_relu',
                    name='combiner'
                )
                combiner.load_weights(combiner_path)
                combiner.summary()

                # Create and load full MoE model
                moe_model = create_mlp_moe(
                    hiddens=hiddens,
                    combiner_hiddens=hiddens,
                    input_dim=n_features,
                    embed_dim=embed_dim,
                    skipped_layers=skipped_layers,
                    skip_repr=skip_repr,
                    pretraining=PRETRAINING_MOE,
                    freeze_experts=FREEZE_EXPERT,
                    expert_paths=expert_paths,
                    mode=MODE_MOE,
                    activation=activation,
                    norm=norm,
                    sam_rho=rho
                )
                moe_model.summary()

                # evaluation of the subset training set and reporting to wandb
                # per each sample, 
                # 3 output values from experts
                # 3 posteriors from combiner
                # 1 weighted sum from the model
                # 1 ground truth label
                # 1 pred vs actual error (expected positive errors)
                # Sort the values by error and sort by largest error to lowest.
                # Get predictions for subset
                expert_plus_preds = expert_plus.predict(X_train_subset)[1]
                expert_zero_preds = expert_zero.predict(X_train_subset)[1]
                expert_minus_preds = expert_minus.predict(X_train_subset)[1]
                combiner_probs = combiner.predict(X_train_subset)[1]
                moe_probs, moe_preds = moe_model.predict(X_train_subset)

                # Squeeze the extra dimension from expert predictions so shapes match
                expert_plus_preds = np.squeeze(expert_plus_preds, axis=-1)
                expert_zero_preds = np.squeeze(expert_zero_preds, axis=-1)
                expert_minus_preds = np.squeeze(expert_minus_preds, axis=-1)

                # Calculate weighted sum (now expert_outputs has shape (batch_size, 3))
                expert_outputs = np.stack([expert_plus_preds, expert_zero_preds, expert_minus_preds], axis=1)
                weighted_sum = np.sum(expert_outputs * combiner_probs, axis=1)

                # Calculate errors and create sorted indices
                errors = np.abs(moe_preds[:, 0] - y_train_subset[:, 0])
                sorted_indices = np.argsort(errors)[::-1]  # Sort in descending order

                # Print results for each sample, sorted by error
                print("\nAnalysis of subset samples (sorted by error, largest to smallest):")
                print("-" * 80)
                for idx in sorted_indices:
                    print(f"\nSample {idx + 1} (Error: {errors[idx]:.4f}):")
                    print(f"Ground truth: {y_train_subset[idx][0]:.4f}")
                    print(f"MoE preds: {moe_preds[idx][0]:.4f}")
                    print(f"Weighted sum: {weighted_sum[idx]:.4f}")
                    print(f"Expert preds: y+:{expert_plus_preds[idx]:.4f}, "
                          f"y0:{expert_zero_preds[idx]:.4f}, "
                          f"y-:{expert_minus_preds[idx]:.4f}")
                    print(f"Combiner probs: p+:{combiner_probs[idx][0]:.4f}, "
                          f"p0:{combiner_probs[idx][1]:.4f}, "
                          f"p-:{combiner_probs[idx][2]:.4f}")
                    print(f"Pred error: {errors[idx]:.4f}")
                print("-" * 80)

                # Evaluate the model error on subset test set
                expert_plus_preds_test = expert_plus.predict(X_test_subset)[1]
                expert_zero_preds_test = expert_zero.predict(X_test_subset)[1]
                expert_minus_preds_test = expert_minus.predict(X_test_subset)[1]
                combiner_probs_test = combiner.predict(X_test_subset)[1]
                moe_probs_test, moe_preds_test = moe_model.predict(X_test_subset)

                # Squeeze the extra dimension from test expert predictions
                expert_plus_preds_test = np.squeeze(expert_plus_preds_test, axis=-1)
                expert_zero_preds_test = np.squeeze(expert_zero_preds_test, axis=-1)
                expert_minus_preds_test = np.squeeze(expert_minus_preds_test, axis=-1)

                # Calculate weighted sum for test set
                expert_outputs_test = np.stack(
                    [expert_plus_preds_test, expert_zero_preds_test, expert_minus_preds_test], axis=1)
                weighted_sum_test = np.sum(expert_outputs_test * combiner_probs_test, axis=1)

                # Calculate errors and create sorted indices for test set
                errors_test = np.abs(moe_preds_test[:, 0] - y_test_subset[:, 0])
                sorted_indices_test = np.argsort(errors_test)[::-1]  # Sort in descending order

                # Print results for each test sample, sorted by error
                print("\nAnalysis of test subset samples (sorted by error, largest to smallest):")
                print("-" * 80)
                for idx in sorted_indices_test:
                    print(f"\nSample {idx + 1} (Error: {errors_test[idx]:.4f}):")
                    print(f"Ground truth: {y_test_subset[idx][0]:.4f}")
                    print(f"MoE preds: {moe_preds_test[idx][0]:.4f}")
                    print(f"Weighted sum: {weighted_sum_test[idx]:.4f}")
                    print(f"Expert preds: y+:{expert_plus_preds_test[idx]:.4f}, "
                          f"y0:{expert_zero_preds_test[idx]:.4f}, "
                          f"y-:{expert_minus_preds_test[idx]:.4f}")
                    print(f"Combiner probs: p+:{combiner_probs_test[idx][0]:.4f}, "
                          f"p0:{combiner_probs_test[idx][1]:.4f}, "
                          f"p-:{combiner_probs_test[idx][2]:.4f}")
                    print(f"Pred error: {errors_test[idx]:.4f}")
                print("-" * 80)



                # evaluate the model error on test set
                error_mae = evaluate_mae(moe_model, X_test, y_test)
                print(f'mae error: {error_mae}')
                wandb.log({"mae": error_mae})

                # evaluate the model error on training set
                error_mae_train = evaluate_mae(moe_model, X_train, y_train)
                print(f'mae error train: {error_mae_train}')
                wandb.log({"train_mae": error_mae_train})

                # evaluate the model correlation on test set
                error_pcc = evaluate_pcc(moe_model, X_test, y_test)
                print(f'pcc error: {error_pcc}')
                wandb.log({"pcc": error_pcc})

                # evaluate the model correlation on training set
                error_pcc_train = evaluate_pcc(moe_model, X_train, y_train)
                print(f'pcc error train: {error_pcc_train}')
                wandb.log({"train_pcc": error_pcc_train})

                # evaluate the model correlation on test set based on logI and logI_prev
                error_pcc_logI = evaluate_pcc(moe_model, X_test, y_test, logI_test, logI_prev_test)
                print(f'pcc error logI: {error_pcc_logI}')
                wandb.log({"pcc_I": error_pcc_logI})

                # evaluate the model correlation on training set based on logI and logI_prev
                error_pcc_logI_train = evaluate_pcc(moe_model, X_train, y_train, logI_train, logI_prev_train)
                print(f'pcc error logI train: {error_pcc_logI_train}')
                wandb.log({"train_pcc_I": error_pcc_logI_train})

                # evaluate the model on test cme_files
                above_threshold = mae_plus_threshold
                # evaluate the model error for rare samples on test set
                error_mae_cond = evaluate_mae(
                    moe_model, X_test, y_test, above_threshold=above_threshold)
                print(f'mae error delta >= {above_threshold} test: {error_mae_cond}')

                wandb.log({"mae+": error_mae_cond})

                # evaluate the model error for rare samples on training set
                error_mae_cond_train = evaluate_mae(
                    moe_model, X_train, y_train, above_threshold=above_threshold)
                print(f'mae error delta >= {above_threshold} train: {error_mae_cond_train}')
                wandb.log({"train_mae+": error_mae_cond_train})

                # evaluate the model correlation for rare samples on test set
                error_pcc_cond = evaluate_pcc(
                    moe_model, X_test, y_test, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond}')
                wandb.log({"pcc+": error_pcc_cond})

                # evaluate the model correlation for rare samples on training set
                error_pcc_cond_train = evaluate_pcc(
                    moe_model, X_train, y_train, above_threshold=above_threshold)
                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_train}')
                wandb.log({"train_pcc+": error_pcc_cond_train})

                # evaluate the model correlation for rare samples on test set based on logI and logI_prev
                error_pcc_cond_logI = evaluate_pcc(
                    moe_model, X_test, y_test, logI_test, logI_prev_test, above_threshold=above_threshold)

                print(f'pcc error delta >= {above_threshold} test: {error_pcc_cond_logI}')
                wandb.log({"pcc+_I": error_pcc_cond_logI})

                # evaluate the model correlation for rare samples on training set based on logI and logI_prev
                error_pcc_cond_logI_train = evaluate_pcc(
                    moe_model, X_train, y_train, logI_train, logI_prev_train, above_threshold=above_threshold)

                print(f'pcc error delta >= {above_threshold} train: {error_pcc_cond_logI_train}')
                wandb.log({"train_pcc+_I": error_pcc_cond_logI_train})

                # Process SEP event files in the specified directory
                test_directory = root_dir + '/testing'
                filenames = process_sep_events(
                    test_directory,
                    moe_model,
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
                    moe_model,
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

                # Finish the wandb run
                wandb.finish()


if __name__ == '__main__':
    main()
