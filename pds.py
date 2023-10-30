import random
from datetime import datetime

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf

from dataload import seploader as sepl
from evaluate.utils import count_above_threshold, plot_tsne_pds
# types for type hinting
from models import modeling

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
    title = 'PDS, with batches'
    print(title)

    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(
        './cme_and_electron/data')

    elevateds, seps = count_above_threshold(shuffled_train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    # combine training and validation
    combined_train_x, combined_train_y = loader.combine(shuffled_train_x, shuffled_train_y, shuffled_val_x,
                                                        shuffled_val_y)

    for batch_size in [32, 64, 128, 256, 300]:  # Replace with the batch sizes you're interested in
        with mlflow.start_run(run_name=f"Batch_Size_{batch_size}"):
            # Automatic logging
            mlflow.tensorflow.autolog()
            # Log the batch size
            mlflow.log_param("batch_size", batch_size)

            mb = modeling.ModelBuilder()

            # create my feature extractor
            feature_extractor = mb.create_model_pds(input_dim=19, feat_dim=9, hiddens=[18])

            # plot the model
            # mb.plot_model(feature_extractor, "pds_stage1")

            # load weights to continue training
            # feature_extractor.load_weights('model_weights_2023-09-28_18-25-47.h5')
            # print('weights loaded successfully!')

            # Generate a timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # training
            Options = {
                'batch_size': batch_size,
                'epochs': 10000,
                'patience': 25,
                'learning_rate': 0.06,
            }
            mlflow.log_param("batch_size", Options['batch_size'])

            # print options used
            print(Options)
            mb.train_pds(feature_extractor,
                         shuffled_train_x, shuffled_train_y,
                         shuffled_val_x, shuffled_val_y,
                         combined_train_x, combined_train_y,
                         learning_rate=Options['learning_rate'],
                         epochs=Options['epochs'],
                         batch_size=Options['batch_size'],
                         patience=Options['patience'], save_tag=timestamp)

            file_path = plot_tsne_pds(feature_extractor, combined_train_x, combined_train_y, title, 'training',
                                      save_tag=timestamp)
            mlflow.log_artifact(file_path)
            file_path = plot_tsne_pds(feature_extractor, shuffled_test_x, shuffled_test_y, title, 'testing',
                                      save_tag=timestamp)
            # Log t-SNE plot
            mlflow.log_artifact(file_path)


if __name__ == '__main__':
    main()
