import random
from datetime import datetime
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from dataload import DenseReweights as dr
from dataload import seploader as sepl
from evaluate import evaluation as eval
from evaluate.utils import count_above_threshold, plot_tsne_extended
from models import modeling
from typing import Optional, List

# Set the tracking URI to a local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_experiment("Low_Batch_Experiments")
mlflow.set_experiment("Default")

# SEEDING
SEED = 42  # seed number 

# Set NumPy seed
np.random.seed(SEED)

# Set TensorFlow seed
tf.random.set_seed(SEED)

# Set random seed
random.seed(SEED)


def load_model_with_weights(model_type: str,
                            weight_path: str,
                            input_dim: int = 19,
                            feat_dim: int = 9,
                            hiddens: Optional[list] = None,
                            output_dim: int = 1) -> tf.keras.Model:
    """
    Load a model of a given type with pre-trained weights.

    :param output_dim:
    :param model_type: The type of the model to load ('features', 'reg', 'dec', 'features_reg_dec', etc.).
    :param weight_path: The path to the saved weights.
    :param input_dim: The input dimension for the model. Default is 19.
    :param feat_dim: The feature dimension for the model. Default is 9.
    :param hiddens: A list of integers specifying the number of hidden units for each hidden layer.
    :return: A loaded model with pre-trained weights.
    """
    if hiddens is None:
        hiddens = [18]
    mb = modeling.ModelBuilder()

    model = None  # will be determined by model type
    if model_type == 'features_reg_dec':
        # Add logic to create this type of model
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=output_dim,
            with_ae=True, with_reg=True)
    elif model_type == 'features_reg':
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=output_dim,
            with_ae=False, with_reg=True)
    elif model_type == 'features_dec':
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=None,
            with_ae=True, with_reg=False)
    else:  # features
        model = mb.create_model_pds(
            input_dim=input_dim,
            feat_dim=feat_dim,
            hiddens=hiddens,
            output_dim=None,
            with_ae=False, with_reg=False)
    # Load weights into the model
    model.load_weights(weight_path)
    print(f"Weights {weight_path} loaded successfully!")

    return model


def main():
    """
    Main function for testing the AI Panther
    :return: None
    """

    # data_path = '/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data'
    data_path = './cme_and_electron/data'
    # check for gpus
    print(tf.config.list_physical_devices('GPU'))
    # Read the CSV file
    loader = sepl.SEPLoader()
    shuffled_train_x, shuffled_train_y, shuffled_val_x, \
        shuffled_val_y, shuffled_test_x, shuffled_test_y = loader.load_from_dir(data_path)

    # combine and get weights
    combined_train_x, combined_train_y = loader.combine(shuffled_train_x, shuffled_train_y, shuffled_val_x,
                                                        shuffled_val_y)
    min_norm_weight = 0.01 / len(combined_train_y)

    # get validation sample weights based on dense weights
    combined_sample_weights = dr.DenseReweights(
        combined_train_x, combined_train_y, alpha=.9, min_norm_weight=min_norm_weight, debug=False).reweights
    train_length = len(shuffled_train_y)
    val_length = len(shuffled_val_y)

    sample_weights = combined_sample_weights[:train_length]
    val_sample_weights = combined_sample_weights[train_length:train_length + val_length]

    elevateds, seps = count_above_threshold(shuffled_train_y)
    print(f'Sub-Training set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_val_y)
    print(f'Validation set: elevated events: {elevateds}  and sep events: {seps}')
    elevateds, seps = count_above_threshold(shuffled_test_y)
    print(f'Test set: elevated events: {elevateds}  and sep events: {seps}')

    for model_type in ['features_reg']: #, 'features_reg', 'features_dec', 'features_reg_dec']:
        weight_path = "./10-29-2023/best_model_weights_2023-10-26_01-59-56.h5"
        for batch_size, freeze in [(292, False), (292, True), (train_length, False), (train_length, True)]:
            title = f'PDS head, {"with" if batch_size == 292 else "without"} batches,\
             {"frozen" if freeze else "fine-tuned"} features'
            print(title)
            with mlflow.start_run(run_name=f"PDS_DL_REG_128_Head_{batch_size}_freeze_{freeze}"):
                # Automatic logging
                mlflow.tensorflow.autolog()
                # Log the batch size
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("freeze features", freeze)
                mlflow.log_param("model_type", model_type)

                mb = modeling.ModelBuilder()
                # load weights to continue training
                feature_extractor = load_model_with_weights(
                    model_type,
                    weight_path,
                )
                # add the regression head with dense weighting
                regressor = mb.add_reg_proj_head(feature_extractor, freeze_features=freeze, pds=True)

                # plot the model
                mb.plot_model(regressor, 'pds_stage2')

                # Generate a timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                # training
                Options = {
                    'batch_size': batch_size,  # len(shuffled_train_x), #768,
                    'epochs': 100000,
                    'patience': 25,
                    'learning_rate': 6e-4,
                }

                # print options used
                print(Options)
                mb.train_reg_head(regressor,
                                  shuffled_train_x, shuffled_train_y,
                                  shuffled_val_x, shuffled_val_y,
                                  combined_train_x, combined_train_y,
                                  sample_weights=sample_weights,
                                  sample_val_weights=val_sample_weights,
                                  sample_train_weights=combined_sample_weights,
                                  learning_rate=Options['learning_rate'],
                                  epochs=Options['epochs'],
                                  batch_size=Options['batch_size'],
                                  patience=Options['patience'], save_tag=timestamp)

                file_path = plot_tsne_extended(regressor, combined_train_x, combined_train_y, title, 'training',
                                               save_tag=timestamp)
                mlflow.log_artifact(file_path)
                file_path = plot_tsne_extended(regressor, shuffled_test_x, shuffled_test_y, title, 'testing',
                                               save_tag=timestamp)
                mlflow.log_artifact(file_path)

                ev = eval.Evaluator()
                metrics = ev.evaluate(regressor, shuffled_test_x, shuffled_test_y, title, threshold=10,
                                      save_tag='test_' + timestamp)
                # Log each metric in the dictionary
                for key, value in metrics.items():
                    if key == 'plot':
                        mlflow.log_artifact(value)  # Log the plot as an artifact
                    else:
                        mlflow.log_metric(key, value)  # Log other items as metrics

                metrics = ev.evaluate(regressor, combined_train_x, combined_train_y, title, threshold=10,
                                      save_tag='training_' + timestamp)
                # Log each metric in the dictionary
                for key, value in metrics.items():
                    if key == 'plot':
                        mlflow.log_artifact(value)  # Log the plot as an artifact
                    else:
                        mlflow.log_metric(key, value)  # Log other items as metrics


if __name__ == '__main__':
    main()
