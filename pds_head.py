import numpy as np
from models import modeling
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
    title = 'PDS, with batches, frozen features'
    print(title)
    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(
        './cme_and_electron/data')

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
    feature_extractor = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])

    # load weights to continue training
    # feature_extractor.load_weights(
    #     './9-28--29-2023/model_weights_2023-09-29_19-44-41.h5')
    # print(
    #     'weights /home1/jmoukpe2016/keras-functional-api/9-28--29-2023/model_weights_2023-09-29_19-44-41.h5 loaded successfully!')

    # add the regression head with dense weighting
    regressor = mb.add_reg_proj_head(feature_extractor, freeze_features=True, pds=True)

    # plot the model
    mb.plot_model(regressor, 'pds_stage2')

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # training
    Options = {
        'batch_size': 768,  # len(shuffled_train_x), #768,
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
