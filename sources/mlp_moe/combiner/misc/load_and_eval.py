from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report, accuracy_score
from tensorflow_addons.optimizers import AdamW

from modules.evaluate.utils import plot_repr_corr_dist, plot_tsne_delta
from modules.reweighting.ImportanceWeighting import exDenseReweightsD
from modules.shared.globals import *
from modules.training.ts_modeling import (
    build_dataset,
    set_seed,
    create_mlp,
    convert_to_onehot_cls,
    plot_confusion_matrix,
    create_metrics_table,
    filter_ds,
    plot_posteriors
)


def main():
    """
    Main function to load and evaluate a trained Combiner model
    :return:
    """

    # Path to pre-trained model weights
    combiner_path = COMBINER_V2_PCC_CE_S2_C04_INVESTIGATION_BS800

    for seed in SEEDS:
        # PARAMS
        inputs_to_use = INPUTS_TO_USE[0]  # Use first input configuration
        outputs_to_use = OUTPUTS_TO_USE
        add_slope = ADD_SLOPE[0]  # Use first add_slope value
        cme_speed_threshold = CME_SPEED_THRESHOLD[0]  # Use first threshold value

        # Join the inputs_to_use list into a string, replace '.' with '_', and join with '-'
        inputs_str = "_".join(input_type.replace('.', '_') for input_type in inputs_to_use)
        
        # Create a unique experiment name with a timestamp
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f'combiner_v3_pcc_ce_s2_c04_bs800_{current_time}'

        set_seed(seed)
        
        hiddens = MLP_HIDDENS  # hidden layers
        bandwidth = BANDWIDTH
        embed_dim = EMBED_DIM
        output_dim = COMBINER_OUTPUT_DIM  # 3 classes for routing
        dropout = DROPOUT
        activation = ACTIVATION
        norm = NORM
        residual = RESIDUAL
        skip_repr = SKIP_REPR
        skipped_layers = SKIPPED_LAYERS
        N = N_FILTERED  # number of samples to keep outside the threshold
        lower_threshold = LOWER_THRESHOLD_MOE  # lower threshold for the delta_p
        upper_threshold = UPPER_THRESHOLD_MOE  # upper threshold for the delta_p
        rho = RHO_MOE_C[0]

        # Initialize wandb
        wandb.init(project="Jan-moe-router-Report", name=experiment_name)

        # set the root directory
        root_dir = DS_PATH
        
        # build the dataset
        X_train, y_train, _, _ = build_dataset(
            root_dir + '/training',
            inputs_to_use=inputs_to_use,
            add_slope=add_slope,
            outputs_to_use=outputs_to_use,
            cme_speed_threshold=cme_speed_threshold,
            shuffle_data=True)

        # Convert y_train to 3 classes based on thresholds
        y_train_classes = convert_to_onehot_cls(y_train, lower_threshold, upper_threshold)

        # Build test set
        X_test, y_test, _, _ = build_dataset(
            root_dir + '/testing',
            inputs_to_use=inputs_to_use,
            add_slope=add_slope,
            outputs_to_use=outputs_to_use,
            cme_speed_threshold=cme_speed_threshold)

        # Convert y_test to 3 classes based on thresholds
        y_test_classes = convert_to_onehot_cls(y_test, lower_threshold, upper_threshold)

        # filtering training and test sets for additional results
        X_train_filtered, y_train_filtered = filter_ds(
            X_train, y_train,
            low_threshold=LOWER_THRESHOLD,  # std threshold for evals
            high_threshold=UPPER_THRESHOLD,  # std threshold for evals
            N=N, seed=seed)
        X_test_filtered, y_test_filtered = filter_ds(
            X_test, y_test,
            low_threshold=LOWER_THRESHOLD,  # std threshold for evals
            high_threshold=UPPER_THRESHOLD,  # std threshold for evals
            N=N, seed=seed)

        # get the number of input features
        n_features = X_train.shape[1]

        # Create combiner model
        combiner_model = create_mlp(
            input_dim=n_features,
            hiddens=hiddens,
            embed_dim=embed_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            norm=norm,
            skip_repr=skip_repr,
            skipped_layers=skipped_layers,
            sam_rho=rho,
            output_activation='norm_relu'
        )

        # Load trained weights
        print(f"Loading model weights from {combiner_path}")
        combiner_model.load_weights(combiner_path)

        # Get predictions
        predictions = combiner_model.predict(X_train)
        y_train_pred = predictions[1]
        predictions = combiner_model.predict(X_test)
        y_test_pred = predictions[1]

        # Convert predictions to class labels
        y_train_pred_classes = np.argmax(y_train_pred, axis=1)
        y_test_pred_classes = np.argmax(y_test_pred, axis=1)
        y_train_true_classes = np.argmax(y_train_classes, axis=1)
        y_test_true_classes = np.argmax(y_test_classes, axis=1)

        # Calculate confusion matrices and create plots
        class_names = ['minus', 'zero', 'plus']

        # Create and save train confusion matrix plot
        train_cm_fig = plot_confusion_matrix(
            y_train_pred_classes,
            y_train_true_classes,
            class_names=class_names,
            title="Training Confusion Matrix",
            xlabel="Actual",
            ylabel="Predicted",
        )

        # Create and save test confusion matrix plot
        test_cm_fig = plot_confusion_matrix(
            y_test_pred_classes,
            y_test_true_classes,
            class_names=class_names,
            title="Test Confusion Matrix",
            xlabel="Actual",
            ylabel="Predicted",
        )

        # Calculate accuracies
        train_accuracy = accuracy_score(y_train_true_classes, y_train_pred_classes)
        test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)

        # Calculate class-specific accuracies
        train_plus_mask = y_train_true_classes == PLUS_INDEX
        train_zero_mask = y_train_true_classes == MID_INDEX
        train_minus_mask = y_train_true_classes == MINUS_INDEX

        test_plus_mask = y_test_true_classes == PLUS_INDEX
        test_zero_mask = y_test_true_classes == MID_INDEX
        test_minus_mask = y_test_true_classes == MINUS_INDEX

        train_plus_accuracy = accuracy_score(y_train_true_classes[train_plus_mask],
                                             y_train_pred_classes[train_plus_mask])
        train_zero_accuracy = accuracy_score(y_train_true_classes[train_zero_mask],
                                             y_train_pred_classes[train_zero_mask])
        train_minus_accuracy = accuracy_score(y_train_true_classes[train_minus_mask],
                                              y_train_pred_classes[train_minus_mask])

        test_plus_accuracy = accuracy_score(y_test_true_classes[test_plus_mask],
                                            y_test_pred_classes[test_plus_mask])
        test_zero_accuracy = accuracy_score(y_test_true_classes[test_zero_mask],
                                            y_test_pred_classes[test_zero_mask])
        test_minus_accuracy = accuracy_score(y_test_true_classes[test_minus_mask],
                                             y_test_pred_classes[test_minus_mask])

        # Get detailed classification reports
        train_report = classification_report(y_train_true_classes, y_train_pred_classes)
        test_report = classification_report(y_test_true_classes, y_test_pred_classes)

        print("\nTraining Results:")
        print(f'Accuracy: {train_accuracy}')
        print("\nClassification Report:")
        print(train_report)

        print("\nTest Results:")
        print(f'Accuracy: {test_accuracy}')
        print("\nClassification Report:")
        print(test_report)

        # Create metric tables as figures
        train_metrics_fig = create_metrics_table(y_train_true_classes, y_train_pred_classes, "Train")
        test_metrics_fig = create_metrics_table(y_test_true_classes, y_test_pred_classes, "Test")

        # Plot posteriors
        train_posteriors_fig = plot_posteriors(y_train_pred, y_train, suptitle="Train Posterior Probabilities vs. Delta")
        test_posteriors_fig = plot_posteriors(y_test_pred, y_test, suptitle="Test Posterior Probabilities vs. Delta")

        # Log results to wandb
        wandb.log({
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train+_accuracy": train_plus_accuracy,
            "train0_accuracy": train_zero_accuracy,
            "train-_accuracy": train_minus_accuracy,
            "test+_accuracy": test_plus_accuracy,
            "test0_accuracy": test_zero_accuracy,
            "test-_accuracy": test_minus_accuracy,
            "train_confusion_matrix": wandb.Image(train_cm_fig),
            "test_confusion_matrix": wandb.Image(test_cm_fig),
            "train_classification_metrics": wandb.Image(train_metrics_fig),
            "test_classification_metrics": wandb.Image(test_metrics_fig),
            "train_posteriors": wandb.Image(train_posteriors_fig),
            "test_posteriors": wandb.Image(test_posteriors_fig)
        })

        # Close all figures to free memory
        plt.close('all')

        # Evaluate the model correlation with colored
        file_path = plot_repr_corr_dist(
            combiner_model,
            X_train_filtered, y_train_filtered,
            "combiner_training",
            model_type='features_cls')
        wandb.log({'representation_correlation_colored_plot_train': wandb.Image(file_path)})

        file_path = plot_repr_corr_dist(
            combiner_model,
            X_test_filtered, y_test_filtered,
            "combiner_test",
            model_type='features_cls')
        wandb.log({'representation_correlation_colored_plot_test': wandb.Image(file_path)})

        # Log t-SNE plots
        stage1_file_path = plot_tsne_delta(
            combiner_model,
            X_train_filtered, y_train_filtered,
            "combiner", 'training',
            model_type='features_cls',
            save_tag=current_time, seed=seed)
        wandb.log({'combiner_tsne_training_plot': wandb.Image(stage1_file_path)})

        stage1_file_path = plot_tsne_delta(
            combiner_model,
            X_test_filtered, y_test_filtered,
            "combiner", 'testing',
            model_type='features_cls',
            save_tag=current_time, seed=seed)
        wandb.log({'combiner_tsne_testing_plot': wandb.Image(stage1_file_path)})

        # Finish the wandb run
        wandb.finish()


if __name__ == '__main__':
    main()
