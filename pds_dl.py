import numpy as np
from models import modeling
import tensorflow as tf
import random
from datetime import datetime
from dataload import seploader as sepl
from dataload import DenseReweights as dr
from evaluate.utils import count_above_threshold, plot_tsne_pds

# SEEDING
SEED = 42  # seed number 

# Set NumPy seed
np.random.seed(SEED)

# Set TensorFlow seed
tf.random.set_seed(SEED)

# Set random seed
random.seed(SEED)


def main():
    """
    Main function for testing the AI Panther
    :return: None
    """
    title = 'PDS, Dense Joint Loss, with batches'
    print(title)

    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(
        './cme_and_electron/data')

    # get validation sample weights based on dense weights
    train_jweights = dr.DenseJointReweights(
        shuffled_train_x, shuffled_train_y, alpha=.9, debug=False)
    sample_joint_weights = train_jweights.jreweights
    sample_joint_weights_indices = train_jweights.jindices

    # print sample_joint_weights_indices, sample_joint_weights
    # print(f'sample_joint_weights_indices: {sample_joint_weights_indices[:2]}')
    # print(f'sample_joint_weights: {sample_joint_weights[:2]}')
    # print(f'size of sample_joint_weights: {len(sample_joint_weights)}')
    # print(f'size of sample_joint_weights indices: {len(sample_joint_weights_indices)}')

    val_jweights = dr.DenseJointReweights(
        shuffled_val_x, shuffled_val_y, alpha=.9, debug=False)
    val_sample_joint_weights = val_jweights.jreweights
    val_sample_joint_weights_indices = val_jweights.jindices

    elevateds, seps = count_above_threshold(shuffled_train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    mb = modeling.ModelBuilder()

    # create my feature extractor
    feature_extractor = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])

    # plot the model
    # # mb.plot_model(feature_extractor, "pds_stage1")

    # load weights to continue training
    # feature_extractor.load_weights('model_weights_2023-09-28_18-25-47.h5')
    # print('weights loaded successfully!')

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # training
    Options = {
        'batch_size': 128,
        'epochs': 10000,
        'patience': 25,
        'learning_rate': 0.06,
    }

    # print options used
    print(Options)
    mb.train_pds_dl(feature_extractor,
                    shuffled_train_x, shuffled_train_y,
                    shuffled_val_x, shuffled_val_y,
                    sample_joint_weights=sample_joint_weights,
                    sample_joint_weights_indices=sample_joint_weights_indices,
                    val_sample_joint_weights=val_sample_joint_weights,
                    val_sample_joint_weights_indices=val_sample_joint_weights_indices,
                    learning_rate=Options['learning_rate'],
                    epochs=Options['epochs'],
                    batch_size=Options['batch_size'],
                    patience=Options['patience'], save_tag=timestamp)

    # combine training and validation
    combined_train_x, combined_train_y = loader.combine(
        shuffled_train_x, shuffled_train_y, shuffled_val_x,
        shuffled_val_y)

    plot_tsne_pds(feature_extractor,
                  combined_train_x, combined_train_y, title, 'training',
                  save_tag=timestamp)

    plot_tsne_pds(feature_extractor,
                  shuffled_test_x, shuffled_test_y, title, 'testing',
                  save_tag=timestamp)


if __name__ == '__main__':
    main()
