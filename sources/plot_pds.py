import numpy as np
import tensorflow as tf
import random
from modules.evaluate.utils import load_and_plot_tsne
import mlflow
import mlflow.tensorflow

# Set the tracking URI to a local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Default")

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
    # title = 'PDS, Dense Joint Loss, 128'
    # title = 'PDS, Dense Joint Loss, Reg, 128'
    # title = 'PDS, Dense Joint Loss, Reg, AE, 128'
    title = 'PDS, Dense Joint Loss, AE, 128'

    print(title)
    # root = "/home1/jmoukpe2016/keras-functional-api"
    root = "."
    model_path = root + "/10-29-2023/best_model_weights_2023-10-26_01-58-45_dl_dec.h5"
    # model_type = "features"
    # model_type = "features_reg_dec"
    # model_type = "features_reg"
    model_type = "features_dec"
    data_dir = root + '/cme_files/fold/fold_1'

    with mlflow.start_run(run_name=f"PDS_Stage1_DL_{model_type}"):
        # Automatic logging
        mlflow.tensorflow.autolog()
        test_plot_path, training_plot_path = load_and_plot_tsne(
            model_path, model_type, title, data_dir, with_head=False)
        mlflow.log_artifact(test_plot_path)
        mlflow.log_artifact(training_plot_path)


if __name__ == '__main__':
    main()
