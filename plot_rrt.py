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
    title = 'rRT, without batches, frozen features'
    print(title)

    model_path = "/home1/jmoukpe2016/keras-functional-api/10-4-2023/extended_model_weights_rrt_stage_2_2023-10-04_00-37-17.h5"
    model_type = "features_reg"
    data_dir = '/home1/jmoukpe2016/keras-functional-api/cme_and_electron/data'

    load_and_plot_tsne(model_path, model_type, title, data_dir)


if __name__ == '__main__':
    main()
