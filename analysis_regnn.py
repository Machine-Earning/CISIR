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
    title = 'Dense Loss, with batches'
    print(title)

    model_path = "/home1/jmoukpe2016/keras-functional-api/10-4-2023/extended_model_weights_reg_nn_2023-10-04_00-35-04.h5"
    model_type = "regular_reg"
    data_dir = '/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data'
    sep_marker = "o"

    load_and_plot_tsne(model_path, model_type, title, sep_marker, data_dir)


if __name__ == '__main__':
    main()
