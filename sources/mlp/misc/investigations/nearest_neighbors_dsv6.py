from datetime import datetime
import os
import numpy as np
import tensorflow as tf

from modules.evaluate.evaluation import find_k_nearest_neighbors
from modules.shared.globals import *
from modules.training.DenseReweights import exDenseReweights
from modules.training.cme_modeling import ModelBuilder
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp)

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

mb = ModelBuilder()


def main() -> None:
    """
    Main function to run the E-MLP model.
    """
    for seed in SEEDS:
        for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
            for cme_speed_threshold in CME_SPEED_THRESHOLD:
                for alpha in [0.5]:
                    for add_slope in ADD_SLOPE:
                        # PARAMS
                        outputs_to_use = OUTPUTS_TO_USE

                        inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
                        title = f'kNN_MLP__{inputs_str}_slope{str(add_slope)}_alpha{alpha:.2f}_CME{cme_speed_threshold}'
                        title = title.replace(' ', '_').replace(':', '_')
                        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

                        tf.random.set_seed(seed)
                        np.random.seed(seed)

                        hiddens = MLP_HIDDENS
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
                        PDS = True
                        FREEZE = True

                        root_dir = DS_PATH

                        X_test, y_test = build_dataset(
                            root_dir + '/testing',
                            inputs_to_use=inputs_to_use,
                            add_slope=add_slope,
                            outputs_to_use=outputs_to_use,
                            cme_speed_threshold=cme_speed_threshold)

                        # third_e_x, third_e_y = build_dataset(
                        #     root_dir + '/testing',
                        #     inputs_to_use=['e4.4'],
                        #     add_slope=add_slope,
                        #     outputs_to_use=outputs_to_use)

                        print(f'X_test.shape: {X_test.shape}')
                        print(f'y_test.shape: {y_test.shape}')

                        # delta_train = y_train[:, 0]
                        delta_test = y_test[:, 0]
                        # print(f'delta_train.shape: {delta_train.shape}')
                        print(f'delta_test.shape: {delta_test.shape}')

                        print(f'rebalancing the validation set...')
                        min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_test)
                        y_test_weights = exDenseReweights(
                            X_test, delta_test,
                            alpha=COMMON_VAL_ALPHA, bw=bandwidth,
                            min_norm_weight=min_norm_weight,
                            debug=False).reweights
                        print(f'validation set rebalanced.')

                        n_features = X_test.shape[1]
                        print(f'n_features: {n_features}')

                        # weight_path = (f"/home1/jmoukpe2016/keras-functional-api/final_model_weights_MLP_e0_5_e4_4_p6_1_p_slopeFalse_alpha0.50_CME0_20240731-075648_reg.h5")
                        weight_path = (f"/home1/jmoukpe2016/keras-functional-api/final_model_weights_MLP_e0_5_e1_8_p_slopeFalse_alpha0.50_CME0_20240731-115408_reg.h5")
                        final_model_sep = create_mlp(
                            input_dim=n_features,
                            hiddens=hiddens,
                            output_dim=output_dim,
                            repr_dim=repr_dim,
                            dropout_rate=dropout,
                            activation=activation,
                            norm=norm,
                            residual=residual,
                            skipped_layers=skipped_layers
                        )

                        # # Recreate the model architecture for final_model_sep
                        # final_model_sep = mb.add_proj_head(
                        #     final_model_sep_stage1,
                        #     output_dim=output_dim,
                        #     freeze_features=FREEZE,
                        #     pds=PDS,
                        #     hiddens=PROJ_HIDDENS,
                        #     dropout_rate=dropout,
                        #     activation=activation,
                        #     norm=norm,
                        #     residual=residual,
                        #     skipped_layers=skipped_layers,
                        #     name='mlp'
                        # )

                        final_model_sep.summary()

                        final_model_sep.load_weights(weight_path)
                        print(f"Model weights are loaded from {weight_path}")

                        # Get model predictions
                        _, predictions = final_model_sep.predict(X_test)

                        # Find and print the k nearest neighbors for large positives in the test set
                        k = 3  # Set the number of nearest neighbors to find

                        res = find_k_nearest_neighbors(X_test, y_test, predictions, k)


if __name__ == '__main__':
    main()
