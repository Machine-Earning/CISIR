import numpy as np

import seploader as sepl
import DenseReweights as dr
import mlflow

# Set the tracking URI to a local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("KDE")


def main():
    """
    Main to test dense reweighting
    """

    loader = sepl.SEPLoader()

    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir('cme_files/fold/fold_1')

    concatenated_x, concatenated_y = loader.combine(train_x, train_y, val_x, val_y, test_x, test_y)
    # get validation sample weights based on dense weights
    for bw in np.arange(0.1, 5, .1):
        # Initialize MLflow tracking
        with mlflow.start_run(run_name="KDE_Scalar") as run:

            # Log the factor
            mlflow.log_param("bandwidth", bw)
            # get the plot
            # Generate a filename in the local directory
            local_filename = f"bandwidth_{bw}.png"
            _ = dr.DenseReweights(concatenated_x, concatenated_y,
                                  alpha=.9, bandwidth=bw,
                                  tag=local_filename,
                                  debug=True).reweights
            # Log the plot
            mlflow.log_artifact(local_filename)


if __name__ == '__main__':
    main()
