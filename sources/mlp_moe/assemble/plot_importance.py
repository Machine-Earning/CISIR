import os
from modules.reweighting.exQtCReweightsD import exDenseReweightsD as qtc
from modules.reweighting.exCosReweightsD import exDenseReweightsD as cos
from modules.reweighting.exDenseReweightsD import exDenseReweightsD as recip

from modules.shared.globals import *
from modules.training.ts_modeling import build_dataset, plot_importance


def main():
    # Build the training dataset
    print("Building the training dataset. Please wait...")

    X_train, y_train, _, _ = build_dataset(
        DS_PATH + '/training',
        inputs_to_use=INPUTS_TO_USE[0],
        add_slope=ADD_SLOPE[0],
        outputs_to_use=OUTPUTS_TO_USE,
        cme_speed_threshold=CME_SPEED_THRESHOLD[0],
        shuffle_data=True
    )

    print("Training dataset built successfully.")

    # Compute delta and prepare reweights
    delta_train = y_train[:, 0]
    alpha_mse = 0.15
    bandwidth = BANDWIDTH
    min_norm_weight = TARGET_MIN_NORM_WEIGHT / len(delta_train)
    mse_train_weights_dict = qtc(
        X_train,
        delta_train,
        alpha=alpha_mse,
        bw=bandwidth,
        min_norm_weight=min_norm_weight,
        debug=False
    ).label_reweight_dict
    print("Weights computed successfully.")

    # create tag based on alpha_mse and the type of importance function
    tag = f"alpha_{alpha_mse}_{('qtc')}_importance"

    # Plot the importance
    plot_importance(mse_train_weights_dict, tag=tag)


    print("Done.")


if __name__ == '__main__':
    main()
