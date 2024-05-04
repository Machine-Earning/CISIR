import os
from datetime import datetime

# Set the environment variable for CUDA (in case it is necessary)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
import wandb
import numpy as np

from modules.evaluate.utils import (
    plot_tsne_delta,
    plot_repr_correlation,
    plot_repr_corr_density,
    plot_repr_corr_colored,
)
from modules.training.cme_modeling import ModelBuilder
from modules.training.ts_modeling import (
    build_dataset,
    create_mlp,
    filter_ds)
from modules.training.utils import get_weight_path

mb = ModelBuilder()

# Define the lookup dictionary
weight_paths = {
    (True,
     0): '/home1/jmoukpe2016/keras-functional-api/overfit_final_model_weights_20240504-000435MLP_e0_5_e1_8_p_slopeTrue_PDS_bs4096_CME0_dsv3_features_128.h5',

    (False,
     0): '/home1/jmoukpe2016/keras-functional-api/overfit_final_model_weights_20240504-002452MLP_e0_5_e1_8_p_slopeFalse_PDS_bs4096_CME0_dsv3_features_128.h5',
    # (True, 500): '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240406-183733MLP_e0_5_e1_8_p_slopeTrue_PDS_bs12000_CME500_features.h5',
    # (False, 500): '/home1/jmoukpe2016/keras-functional-api/final_model_weights_20240406-200720MLP_e0_5_e1_8_p_slopeFalse_PDS_bs12000_CME500_features.h5',
}


def main():
    """
    Main function to run the E-MLP model
    :return:
    """

    for inputs_to_use in [['e0.5', 'e1.8', 'p']]:
        for add_slope in [True, False]:
            for cme_speed_threshold in [0]:
                for alpha in [0]:
                    # PARAMS
                    # inputs_to_use = ['e0.5']
                    # add_slope = True
                    outputs_to_use = ['delta_p']
                    # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
                    inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)

                    # Construct the title
                    title = f'MLP_PDS_Stage1_{inputs_str}_slope{str(add_slope)}_alpha{alpha:.2f}_CME{cme_speed_threshold}'

                    # Replace any other characters that are not suitable for filenames (if any)
                    title = title.replace(' ', '_').replace(':', '_')

                    # Create a unique experiment name with a timestamp
                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                    experiment_name = f'{title}_{current_time}'

                    seed = 456789
                    tf.random.set_seed(seed)
                    np.random.seed(seed)
                    learning_rate = 1e-2  # og learning rate

                    weight_decay = 1e-8  # higher weight decay
                    momentum_beta1 = 0.9  # higher momentum beta1
                    batch_size = 4096
                    epochs = 25000  # higher epochs
                    hiddens = [
                        2048, 1024,
                        2048, 1024,
                        1024, 512,
                        1024, 512,
                        512, 256,
                        512, 256,
                        256, 128,
                        256, 128,
                        256, 128,
                        128, 128,
                        128, 128,
                        128, 128
                    ]

                    hiddens_str = (", ".join(map(str, hiddens))).replace(', ', '_')
                    loss_key = 'mse'
                    target_change = ('delta_p' in outputs_to_use)
                    # print_batch_mse_cb = PrintBatchMSE()
                    rebalacing = True
                    alpha_rw = alpha
                    bandwidth = 4.42e-2  # 0.0519
                    repr_dim = 128
                    dropout = 0.5
                    activation = None
                    norm = 'batch_norm'
                    pds = True
                    cme_speed_threshold = cme_speed_threshold
                    weight_path = get_weight_path(weight_paths, add_slope, cme_speed_threshold)
                    residual = True
                    skipped_layers = 2
                    N = 1000
                    lower_threshold = -0.5
                    upper_threshold = 0.5
                    # Initialize wandb
                    wandb.init(project="nasa-ts-delta-overfit", name=experiment_name, config={
                        "inputs_to_use": inputs_to_use,
                        "add_slope": add_slope,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "momentum_beta1": momentum_beta1,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        # hidden in a more readable format  (wandb does not support lists)
                        "hiddens": hiddens_str,
                        "loss": loss_key,
                        "target_change": target_change,
                        "printing_batch_mse": False,
                        "seed": seed,
                        "rebalancing": rebalacing,
                        "alpha_rw": alpha_rw,
                        "bandwidth": bandwidth,
                        "reciprocal_reweight": True,
                        "repr_dim": repr_dim,
                        "dropout": dropout,
                        "activation": 'LeakyReLU',
                        "norm": norm,
                        'optimizer': 'adam',
                        'architecture': 'mlp',
                        "pds": pds,
                        "stage": 2,
                        "stage1_weights": weight_path,
                        "cme_speed_threshold": cme_speed_threshold,
                        "residual": residual,
                        "skipped_layers": skipped_layers,
                        "ds_version": 5
                    })

                    # set the root directory
                    root_dir = 'data/electron_cme_data_split_v5'
                    # build the dataset
                    X_train, y_train = build_dataset(
                        root_dir + '/training',
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        cme_speed_threshold=cme_speed_threshold)

                    X_train_filtered, y_train_filtered = filter_ds(
                        X_train, y_train,
                        low_threshold=lower_threshold,
                        high_threshold=upper_threshold,
                        N=N, seed=seed)

                    X_test, y_test = build_dataset(
                        root_dir + '/testing',
                        inputs_to_use=inputs_to_use,
                        add_slope=add_slope,
                        outputs_to_use=outputs_to_use,
                        cme_speed_threshold=cme_speed_threshold)

                    X_test_filtered, y_test_filtered = filter_ds(
                        X_test, y_test,
                        low_threshold=lower_threshold,
                        high_threshold=upper_threshold,
                        N=N, seed=seed)

                    # print all cme_files shapes
                    print(f'X_train.shape: {X_train.shape}')
                    print(f'y_train.shape: {y_train.shape}')
                    print(f'X_train_filtered.shape: {X_train_filtered.shape}')
                    print(f'y_train_filtered.shape: {y_train_filtered.shape}')
                    print(f'X_test.shape: {X_test.shape}')
                    print(f'y_test.shape: {y_test.shape}')
                    print(f'X_test_filtered.shape: {X_test_filtered.shape}')
                    print(f'y_test_filtered.shape: {y_test_filtered.shape}')

                    # get the number of features
                    n_features = X_train.shape[1]
                    print(f'n_features: {n_features}')

                    # create the model
                    model_sep_stage1 = create_mlp(
                        input_dim=n_features,
                        hiddens=hiddens,
                        output_dim=0,
                        pds=pds,
                        repr_dim=repr_dim,
                        dropout_rate=dropout,
                        activation=activation,
                        norm=norm,
                        residual=residual,
                        skipped_layers=skipped_layers
                    )
                    model_sep_stage1.summary()

                    # load the weights from the first stage
                    print(f'weights loading from: {weight_path}')
                    model_sep_stage1.load_weights(weight_path)
                    # print the save
                    print(f'weights loaded successfully from: {weight_path}')

                    ## Evalute the model correlation
                    file_path = plot_repr_correlation(
                        model_sep_stage1,
                        X_train_filtered, y_train_filtered,
                        title + "_training"
                    )
                    wandb.log({'representation_correlation_plot_train': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    file_path = plot_repr_correlation(
                        model_sep_stage1,
                        X_test_filtered, y_test_filtered,
                        title + "_test"
                    )
                    wandb.log({'representation_correlation_plot_test': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    ## Evalute the model correlation density
                    file_path = plot_repr_corr_density(
                        model_sep_stage1,
                        X_train_filtered, y_train_filtered,
                        title + "_training"
                    )
                    wandb.log({'representation_correlation_density_plot_train': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    file_path = plot_repr_corr_density(
                        model_sep_stage1,
                        X_test_filtered, y_test_filtered,
                        title + "_test"
                    )
                    wandb.log({'representation_correlation_density_plot_test': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    ## Evalute the model correlation with colored
                    file_path = plot_repr_corr_colored(
                        model_sep_stage1,
                        X_train_filtered, y_train_filtered,
                        title + "_training"
                    )
                    wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    file_path = plot_repr_corr_colored(
                        model_sep_stage1,
                        X_test_filtered, y_test_filtered,
                        title + "_test"
                    )
                    wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})
                    print('file_path: ' + file_path)

                    ## Log t-SNE plot
                    # Log the training t-SNE plot to wandb
                    stage1_file_path = plot_tsne_delta(
                        model_sep_stage1,
                        X_train_filtered, y_train_filtered, title,
                        'stage1_training',
                        model_type='features',
                        save_tag=current_time, seed=seed)
                    wandb.log({'stage1_tsne_training_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Log the testing t-SNE plot to wandb
                    stage1_file_path = plot_tsne_delta(
                        model_sep_stage1,
                        X_test_filtered, y_test_filtered, title,
                        'stage1_testing',
                        model_type='features',
                        save_tag=current_time, seed=seed)
                    wandb.log({'stage1_tsne_testing_plot': wandb.Image(stage1_file_path)})
                    print('stage1_file_path: ' + stage1_file_path)

                    # Finish the wandb run
                    wandb.finish()


if __name__ == '__main__':
    main()
