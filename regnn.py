import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# types for type hinting
from models import modeling
from sklearn.manifold import TSNE
import tensorflow as tf
import random
from datetime import datetime
from dataload import DenseReweights as dr
from evaluate import evaluation as eval
from dataload import seploader as sepl
from evaluate.utils import count_above_threshold, plot_tsne_extended

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

    title = 'DenseLoss, with batches'
    print(title)

    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(
        '/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data')

    # get validation sample weights based on dense weights
    sample_weights = dr.DenseReweights(shuffled_train_x, shuffled_train_y, alpha=.9, debug=False).reweights
    val_sample_weights = dr.DenseReweights(shuffled_val_x, shuffled_val_y, alpha=.9, debug=False).reweights

    elevateds, seps = count_above_threshold(shuffled_train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    mb = modeling.ModelBuilder()

    # create my feature extractor
    regressor = mb.create_model(input_dim=19, feat_dim=9, output_dim=1, hiddens=[18])

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # training

    # combine training and validation
    combined_train_x, combined_train_y = loader.combine(shuffled_train_x, shuffled_train_y, shuffled_val_x,
                                                        shuffled_val_y)

    # training
    Options = {
        'batch_size': 768,
        'epochs': 100000,
        'patience': 25,
        'learning_rate': 3e-4,
    }

    # print options used
    print(Options)
    mb.train_reg_head(regressor, shuffled_train_x, shuffled_train_y, shuffled_val_x, shuffled_val_y,
                      sample_weights=sample_weights, sample_val_weights=val_sample_weights,
                      learning_rate=Options['learning_rate'],
                      epochs=Options['epochs'],
                      batch_size=Options['batch_size'],
                      patience=Options['patience'], save_tag='reg_nn_' + timestamp)

    plot_tsne_extended(regressor, combined_train_x, combined_train_y, title, 'reg_nn_training_',
                                save_tag=timestamp)

    plot_tsne_extended(regressor, shuffled_test_x, shuffled_test_y, title, 'reg_nn_testing_',
                                save_tag=timestamp)

    ev = eval.Evaluator()
    ev.evaluate(regressor, shuffled_test_x, shuffled_test_y, title, threshold=10, save_tag='reg_nn_test_' + timestamp)
    # ev.evaluate(regressor, shuffled_test_x, shuffled_test_y, threshold=1, save_tag='test_' + timestamp)

    ev.evaluate(regressor, combined_train_x, combined_train_y, title, threshold=10,
                save_tag='reg_nn_training_' + timestamp)
    # ev.evaluate(regressor, shuffled_train_x, shuffled_train_y, threshold=1, save_tag='training_' + timestamp)


if __name__ == '__main__':
    main()
