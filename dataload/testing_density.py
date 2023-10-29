import seploader as sepl
import DenseReweights as dr


def main():
    """
    Main to test dense reweighting
    """
    loader = sepl.SEPLoader()

    train_x, train_y, val_x, val_y, test_x, test_y = loader.load_from_dir('cme_and_electron/data')

    concatenated_x, concatenated_y = loader.combine(train_x, train_y, val_x, val_y, test_x, test_y)
    # get validation sample weights based on dense weights
    _ = dr.DenseReweights(concatenated_x, concatenated_y, alpha=.9, debug=True).reweights


if __name__ == '__main__':
    main()
