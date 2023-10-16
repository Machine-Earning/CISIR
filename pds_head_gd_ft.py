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
    title = 'PDS, without batches, fine-tuned features'
    print(title)
    # check for gpus
    tf.config.list_physical_devices('GPU')
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(
        '/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data')

    # get validation sample weights based on dense weights
    sample_weights = dr.DenseReweights(shuffled_train_x, shuffled_train_y, alpha=.9, debug=False).reweights
    val_sample_weights = dr.DenseReweights(shuffled_val_x, shuffled_val_y, alpha=.9, debug=False).reweights

    train_count = count_above_threshold(shuffled_train_y)
    val_count = count_above_threshold(shuffled_val_y)
    test_count = count_above_threshold(shuffled_test_y)

    print(f'Training set: {train_count} above the threshold')
    print(f'Validation set: {val_count} above the threshold')
    print(f'Test set: {test_count} above the threshold')

    mb = modeling.ModelBuilder()

    # create my feature extractor
    feature_extractor = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])

    # load weights to continue training
    feature_extractor.load_weights(
        '/home1/jmoukpe2016/keras-functional-api/model_weights_2023-10-04_08-54-53.h5')
    print(
        'weights /home1/jmoukpe2016/keras-functional-api/model_weights_2023-10-04_08-54-53.h5 loaded successfully!')

    # add the regression head with dense weighting
    regressor = mb.add_reg_proj_head(feature_extractor, freeze_features=False, pds=True)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # training
    Options = {
        'batch_size': len(shuffled_train_y),
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
                      patience=Options['patience'], save_tag=timestamp)

    # combine training and validation
    combined_train_x, combined_train_y = loader.combine(shuffled_train_x, shuffled_train_y, shuffled_val_x,
                                                        shuffled_val_y)

    plot_tsne_extended(regressor, combined_train_x, combined_train_y, title, 'training',
                                save_tag=timestamp)

    plot_tsne_extended(regressor, shuffled_test_x, shuffled_test_y, title, 'testing',
                                save_tag=timestamp)

    ev = eval.Evaluator()
    ev.evaluate(regressor, shuffled_test_x, shuffled_test_y, title, threshold=10, save_tag='test_' + timestamp)
    # ev.evaluate(regressor, shuffled_test_x, shuffled_test_y, threshold=1, save_tag='test_' + timestamp)

    ev.evaluate(regressor, combined_train_x, combined_train_y, title, threshold=10, save_tag='training_' + timestamp)
    # ev.evaluate(regressor, shuffled_train_x, shuffled_train_y, threshold=1, save_tag='training_' + timestamp)


if __name__ == '__main__':
    main()
