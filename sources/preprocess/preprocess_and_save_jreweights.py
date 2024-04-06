import os
import random
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.callbacks import ReduceLROnPlateau
from wandb.keras import WandbCallback

from modules.evaluate.utils import plot_tsne_delta, plot_repr_correlation
from modules.training import cme_modeling
from modules.training.ts_modeling import build_dataset, create_mlp, reshape_X
from modules.reweighting.exDenseReweightsD import exDenseJointReweightsGPU

from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Set up mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# SEEDING
SEED = 456789  # seed number

# Set NumPy seed
np.random.seed(SEED)

# Set TensorFlow seed
tf.random.set_seed(SEED)

# Set random seed
random.seed(SEED)

mb = cme_modeling.ModelBuilder()


def main():
    """
    Main function to run the PDS model
    :return:
    """
    # list the devices available
    devices = tf.config.list_physical_devices('GPU')
    print(f'devices: {devices}')
    # Define the dataset options, including the sharding policy
    for alpha in np.arange(0, 1, 0.1):
        # PARAMS
        inputs_to_use = ['e0.5', 'e1.8', 'p']
        add_slope = True
        outputs_to_use = ['delta_p']
        bandwidth = 0.099
        alpha_rw = alpha

        # set the root directory
        root_dir = "data/electron_cme_data_split"
        # build the dataset
        X_train, y_train = build_dataset(root_dir + '/training',
                                         inputs_to_use=inputs_to_use,
                                         add_slope=add_slope,
                                         outputs_to_use=outputs_to_use)
        X_subtrain, y_subtrain = build_dataset(root_dir + '/subtraining',
                                               inputs_to_use=inputs_to_use,
                                               add_slope=add_slope,
                                               outputs_to_use=outputs_to_use)
        X_test, y_test = build_dataset(root_dir + '/testing',
                                       inputs_to_use=inputs_to_use,
                                       add_slope=add_slope,
                                       outputs_to_use=outputs_to_use)
        X_val, y_val = build_dataset(root_dir + '/validation',
                                     inputs_to_use=inputs_to_use,
                                     add_slope=add_slope,
                                     outputs_to_use=outputs_to_use)

        # print all cme_files shapes
        print(f'X_train.shape: {X_train.shape}')
        print(f'y_train.shape: {y_train.shape}')
        print(f'X_subtrain.shape: {X_subtrain.shape}')
        print(f'y_subtrain.shape: {y_subtrain.shape}')
        print(f'X_test.shape: {X_test.shape}')
        print(f'y_test.shape: {y_test.shape}')
        print(f'X_val.shape: {X_val.shape}')
        print(f'y_val.shape: {y_val.shape}')

        # get the number of features
        n_features = X_train.shape[1]
        print(f'n_features: {n_features}')

        # Compute the sample weights
        delta_train = y_train[:, 0]
        delta_subtrain = y_subtrain[:, 0]
        print(f'delta_train.shape: {delta_train.shape}')
        print(f'delta_subtrain.shape: {delta_subtrain.shape}')

        print(f'rebalancing the training set...')
        min_norm_weight = 0.01 / len(delta_train)

        train_jweights = exDenseJointReweightsGPU(
            X_train, delta_train,
            alpha=alpha_rw, bw=bandwidth,
            min_norm_weight=min_norm_weight, debug=False)

        # The train_jweights object contains the combined joint reweighting info
        train_sample_joint_weights = train_jweights.jreweights
        train_sample_joint_weights_indices = train_jweights.jindices
        print(f'done rebalancing the training set...')

        print(f'rebalancing the subtraining set...')
        min_norm_weight = 0.01 / len(delta_subtrain)

        subtrain_jweights = exDenseJointReweightsGPU(
            X_subtrain, delta_subtrain,
            alpha=alpha_rw, bw=bandwidth,
            min_norm_weight=min_norm_weight, debug=False)

        # The subtrain_jweights object contains the combined joint reweighting info
        subtrain_sample_joint_weights = subtrain_jweights.jreweights
        subtrain_sample_joint_weights_indices = subtrain_jweights.jindices

        print(f'done rebalancing the subtraining set...')
        # save the file to disk




if __name__ == '__main__':
    main()
