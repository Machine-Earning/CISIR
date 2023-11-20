import os
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

# # Set the tracking URI to a local directory
# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment("Low_Batch_Experiments")

# Set the DagsHub credentials programmatically
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ERUD1T3'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '0b7739bcc448e3336dcc7437b505c44cc1801f9c'

# Configure MLflow to connect to DagsHub
mlflow.set_tracking_uri('https://dagshub.com/ERUD1T3/keras-functional-api.mlflow')
mlflow.set_experiment("low_batch_exps_aip")

# List of seeds for multiple runs
seeds = [42, 1000]
# seeds = [42]


def main():
    """
    Main function for testing the AI Panther
    :return: None
    """
    data_path = '/home1/jmoukpe2016/keras-functional-api/cme_and_electron/folds/fold_1'
    # data_path = './cme_and_electron/folds/fold_1'

    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(data_path)

    train_length = len(shuffled_train_y)
    # val_length = len(shuffled_val_y)

    elevateds, seps = count_above_threshold(shuffled_train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    # combine training and validation
    combined_train_x, combined_train_y = loader.combine(shuffled_train_x, shuffled_train_y, shuffled_val_x,
                                                        shuffled_val_y)

    for seed in seeds:
        # Set the seeds for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        for batch_size in [4, -1]:  # Replace with the batch sizes you're interested in
            title = f'PDS, batche size {batch_size}, seed {seed}'
            print(title)
            with mlflow.start_run(run_name=f"PDS_{batch_size}_Seed_{seed}"):
                # Automatic logging
                # mlflow.tensorflow.autolog()
                # Log the batch size and the seed
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("seed", seed)

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
                    'learning_rate': 0.05,
                    'timestamp': timestamp
                }

                # print options used
                print(Options)
                mlflow.log_params(Options)
                # reset the counts
                # mb.sep_sep_count = 0
                # mb.sep_elevated_count = 0
                # mb.sep_background_count = 0
                # mb.elevated_elevated_count = 0
                # mb.elevated_background_count = 0
                # mb.background_background_count = 0
                # mb.number_of_batches = 0

                _, entire_training_loss = mb.investigate_pds(feature_extractor,
                                                             shuffled_train_x, shuffled_train_y,
                                                             shuffled_val_x, shuffled_val_y,
                                                             combined_train_x, combined_train_y,
                                                             learning_rate=Options['learning_rate'],
                                                             epochs=Options['epochs'],
                                                             batch_size=Options['batch_size'],
                                                             patience=Options['patience'],
                                                             save_tag=timestamp + f"_features_{batch_size}")

                # Log the training loss to MLflow
                mlflow.log_metric("entire_train_loss", entire_training_loss)

                sep_sep_count = int(mb.sep_sep_count.numpy())
                sep_elevated_count = int(mb.sep_elevated_count.numpy())
                sep_background_count = int(mb.sep_background_count.numpy())
                elevated_elevated_count = int(mb.elevated_elevated_count.numpy())
                elevated_background_count = int(mb.elevated_background_count.numpy())
                background_background_count = int(mb.background_background_count.numpy())

                mlflow.log_metric("sep_sep", sep_sep_count)
                mlflow.log_metric("sep_elevated", sep_elevated_count)
                mlflow.log_metric("sep_background", sep_background_count)
                mlflow.log_metric("elevated_elevated", elevated_elevated_count)
                mlflow.log_metric("elevated_background", elevated_background_count)
                mlflow.log_metric("background_background", background_background_count)

                total_pairs = (sep_sep_count + sep_elevated_count + sep_background_count +
                               elevated_elevated_count + elevated_background_count +
                               background_background_count)

                mlflow.log_metric("total_pairs", total_pairs)
                mlflow.log_metric("number_of_batches", mb.number_of_batches)

                percent_sep_sep = (sep_sep_count / total_pairs) * 100 if total_pairs > 0 else 0
                mlflow.log_metric("percent_sep_sep", percent_sep_sep)

                file_path = plot_tsne_pds(feature_extractor,
                                          combined_train_x,
                                          combined_train_y,
                                          title, 'training',
                                          save_tag=timestamp,
                                          seed=seed)
                mlflow.log_artifact(file_path)
                print('file_path' + file_path)
                file_path = plot_tsne_pds(feature_extractor,
                                          shuffled_test_x,
                                          shuffled_test_y,
                                          title, 'testing',
                                          save_tag=timestamp,
                                          seed=seed)
                # Log t-SNE plot
                mlflow.log_artifact(file_path)
                print('file_path' + file_path)


if __name__ == '__main__':
    main()
