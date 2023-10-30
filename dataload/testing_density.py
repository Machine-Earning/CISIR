import numpy as np

import seploader as sepl
import DenseReweights as dr
import mlflow


def main():
    """
    Main to test dense reweighting
    """

    loader = sepl.SEPLoader()

    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir('cme_and_electron/data')

    concatenated_x, concatenated_y = loader.combine(train_x, train_y, val_x, val_y, test_x, test_y)
    # get validation sample weights based on dense weights
    for factor in np.arange(0.1, 5, .1):
        # Initialize MLflow tracking
        with mlflow.start_run(run_name="KDE_Analysis") as run:
            existing_run_id = run.info.run_id

            # Log the factor
            mlflow.log_param("bandwidth_factor", factor)
            # get the plot
            # Generate a filename in the local directory
            local_filename = f"kde_plot_factor_{factor}.png"
            _ = dr.DenseReweights(concatenated_x, concatenated_y, alpha=.9, bw_factor=factor, tag=local_filename,
                                  runid=existing_run_id,
                                  debug=True).reweights
            # Log the plot
            mlflow.log_artifact(local_filename)


if __name__ == '__main__':
    main()
