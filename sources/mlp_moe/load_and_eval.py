import os
from datetime import datetime

import numpy as np
import wandb

from modules.shared.globals import *
from modules.training.ts_modeling import (
    build_dataset,
    evaluate_mae,
    evaluate_pcc,
    get_plus_cls,
    get_zero_cls,
    get_minus_cls,
    create_mlp,
)

# Model paths
one_model = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse1.00_v8_updated_20241120-180201_reg.h5'
pos_expert = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse0.10_plus_e_20241212-140850_reg.h5'
neg_expert = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse0.03_minus_e_20241212-133248_reg.h5'
nz_expert = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse0.10_zero_e_20241205-111054_reg.h5'

def main():
    """
    Main function to load and evaluate the E-MLP model
    """
    for seed in [456789]:
        for inputs_to_use in INPUTS_TO_USE:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha_mse, alphaV_mse, alpha_pcc, alphaV_pcc in REWEIGHTS:
                    for rho in RHO:
                        for add_slope in ADD_SLOPE:
                            # Basic setup
                            outputs_to_use = OUTPUTS_TO_USE
                            title = f'mlp2_amse{alpha_mse:.2f}'
                            title = title.replace(' ', '_').replace(':', '_')
                            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                            experiment_name = f'{title}_{current_time}'

                            # Model parameters
                            hiddens = MLP_HIDDENS
                            hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                            embed_dim = EMBED_DIM
                            output_dim = len(outputs_to_use)
                            dropout = DROPOUT
                            activation = ACTIVATION
                            norm = NORM
                            skip_repr = SKIP_REPR
                            skipped_layers = SKIPPED_LAYERS

                            # Initialize wandb
                            wandb.init(project="Jan-Report-Evals", name=experiment_name + "_eval")

                            # Build datasets
                            root_dir = DS_PATH
                            X_train, y_train, logI_train, logI_prev_train = build_dataset(
                                root_dir + '/training',
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                cme_speed_threshold=cme_speed_threshold,
                                shuffle_data=True)

                            X_test, y_test, logI_test, logI_prev_test = build_dataset(
                                root_dir + '/testing',
                                inputs_to_use=inputs_to_use,
                                add_slope=add_slope,
                                outputs_to_use=outputs_to_use,
                                cme_speed_threshold=cme_speed_threshold)

                            n_features = X_train.shape[1]

                            # Get data subsets for each range
                            X_train_pos, y_train_pos, logI_train_pos, logI_prev_train_pos = get_plus_cls(X_train, y_train, UPPER_THRESHOLD_MOE, logI_train, logI_prev_train)
                            X_train_nz, y_train_nz, logI_train_nz, logI_prev_train_nz = get_zero_cls(X_train, y_train, LOWER_THRESHOLD_MOE, UPPER_THRESHOLD_MOE, logI_train, logI_prev_train)
                            X_train_neg, y_train_neg, logI_train_neg, logI_prev_train_neg = get_minus_cls(X_train, y_train, LOWER_THRESHOLD_MOE, logI_train, logI_prev_train)

                            X_test_pos, y_test_pos, logI_test_pos, logI_prev_test_pos = get_plus_cls(X_test, y_test, UPPER_THRESHOLD_MOE, logI_test, logI_prev_test)
                            X_test_nz, y_test_nz, logI_test_nz, logI_prev_test_nz = get_zero_cls(X_test, y_test, LOWER_THRESHOLD_MOE, UPPER_THRESHOLD_MOE, logI_test, logI_prev_test)
                            X_test_neg, y_test_neg, logI_test_neg, logI_prev_test_neg = get_minus_cls(X_test, y_test, LOWER_THRESHOLD_MOE, logI_test, logI_prev_test)

                            # Create model architecture
                            model = create_mlp(
                                input_dim=n_features,
                                hiddens=hiddens,
                                embed_dim=embed_dim,
                                output_dim=output_dim,
                                dropout=dropout,
                                activation=activation,
                                norm=norm,
                                skip_repr=skip_repr,
                                skipped_layers=skipped_layers,
                                sam_rho=rho
                            )

                            # Evaluate one model on all ranges
                            model.load_weights(one_model)
                            
                            # Positive range
                            wandb.log({
                                "one_model_pos_mae_train": evaluate_mae(model, X_train_pos, y_train_pos),
                                "one_model_pos_mae_test": evaluate_mae(model, X_test_pos, y_test_pos),
                                "one_model_pos_pcc_train": evaluate_pcc(model, X_train_pos, y_train_pos),
                                "one_model_pos_pcc_test": evaluate_pcc(model, X_test_pos, y_test_pos),
                                "one_model_pos_pcc_I_train": evaluate_pcc(model, X_train_pos, y_train_pos, logI_train_pos, logI_prev_train_pos),
                                "one_model_pos_pcc_I_test": evaluate_pcc(model, X_test_pos, y_test_pos, logI_test_pos, logI_prev_test_pos)
                            })

                            # Near-zero range
                            wandb.log({
                                "one_model_nz_mae_train": evaluate_mae(model, X_train_nz, y_train_nz),
                                "one_model_nz_mae_test": evaluate_mae(model, X_test_nz, y_test_nz),
                                "one_model_nz_pcc_train": evaluate_pcc(model, X_train_nz, y_train_nz),
                                "one_model_nz_pcc_test": evaluate_pcc(model, X_test_nz, y_test_nz),
                                "one_model_nz_pcc_I_train": evaluate_pcc(model, X_train_nz, y_train_nz, logI_train_nz, logI_prev_train_nz),
                                "one_model_nz_pcc_I_test": evaluate_pcc(model, X_test_nz, y_test_nz, logI_test_nz, logI_prev_test_nz)
                            })

                            # Negative range
                            wandb.log({
                                "one_model_neg_mae_train": evaluate_mae(model, X_train_neg, y_train_neg),
                                "one_model_neg_mae_test": evaluate_mae(model, X_test_neg, y_test_neg),
                                "one_model_neg_pcc_train": evaluate_pcc(model, X_train_neg, y_train_neg),
                                "one_model_neg_pcc_test": evaluate_pcc(model, X_test_neg, y_test_neg),
                                "one_model_neg_pcc_I_train": evaluate_pcc(model, X_train_neg, y_train_neg, logI_train_neg, logI_prev_train_neg),
                                "one_model_neg_pcc_I_test": evaluate_pcc(model, X_test_neg, y_test_neg, logI_test_neg, logI_prev_test_neg)
                            })

                            # Evaluate positive expert on positive range
                            model.load_weights(pos_expert)
                            wandb.log({
                                "pos_expert_mae_train": evaluate_mae(model, X_train_pos, y_train_pos),
                                "pos_expert_mae_test": evaluate_mae(model, X_test_pos, y_test_pos),
                                "pos_expert_pcc_train": evaluate_pcc(model, X_train_pos, y_train_pos),
                                "pos_expert_pcc_test": evaluate_pcc(model, X_test_pos, y_test_pos),
                                "pos_expert_pcc_I_train": evaluate_pcc(model, X_train_pos, y_train_pos, logI_train_pos, logI_prev_train_pos),
                                "pos_expert_pcc_I_test": evaluate_pcc(model, X_test_pos, y_test_pos, logI_test_pos, logI_prev_test_pos)
                            })

                            # Evaluate negative expert on negative range
                            model.load_weights(neg_expert)
                            wandb.log({
                                "neg_expert_mae_train": evaluate_mae(model, X_train_neg, y_train_neg),
                                "neg_expert_mae_test": evaluate_mae(model, X_test_neg, y_test_neg),
                                "neg_expert_pcc_train": evaluate_pcc(model, X_train_neg, y_train_neg),
                                "neg_expert_pcc_test": evaluate_pcc(model, X_test_neg, y_test_neg),
                                "neg_expert_pcc_I_train": evaluate_pcc(model, X_train_neg, y_train_neg, logI_train_neg, logI_prev_train_neg),
                                "neg_expert_pcc_I_test": evaluate_pcc(model, X_test_neg, y_test_neg, logI_test_neg, logI_prev_test_neg)
                            })

                            # Evaluate near-zero expert on near-zero range
                            model.load_weights(nz_expert)
                            wandb.log({
                                "nz_expert_mae_train": evaluate_mae(model, X_train_nz, y_train_nz),
                                "nz_expert_mae_test": evaluate_mae(model, X_test_nz, y_test_nz),
                                "nz_expert_pcc_train": evaluate_pcc(model, X_train_nz, y_train_nz),
                                "nz_expert_pcc_test": evaluate_pcc(model, X_test_nz, y_test_nz),
                                "nz_expert_pcc_I_train": evaluate_pcc(model, X_train_nz, y_train_nz, logI_train_nz, logI_prev_train_nz),
                                "nz_expert_pcc_I_test": evaluate_pcc(model, X_test_nz, y_test_nz, logI_test_nz, logI_prev_test_nz)
                            })

                            wandb.finish()


if __name__ == '__main__':
    main()
