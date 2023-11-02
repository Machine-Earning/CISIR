import numpy as np
import tensorflow as tf
import random
from evaluate.utils import load_and_plot_tsne

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
    title = 'PDS, with batches, fine-tuned features'
    print(title)
    # root = "/home1/jmoukpe2016/keras-functional-api"
    root = "."
    model_path = root + "/best_model_weights_2023-10-30_22-57-59_features.h5"
    model_type = "features"
    data_dir = root + '/cme_and_electron/data'
    sep_marker = "x"

    load_and_plot_tsne(model_path, model_type, title, sep_marker, data_dir, with_head=False)


if __name__ == '__main__':
    main()
