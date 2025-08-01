# Dataset configurations
DS_VERSION = 1  # Dataset version
DS_PATH = 'data/sep_cme'  # Path to the dataset

OUTPUT_DIM = 1


# Training configurations
TRIAL_SEEDS = [456789, 42, 123, 0, 9999] # Seeds for trial
TRIAL_SEED = [456789]
BATCH_SIZE = 200  # Batch size
EPOCHS = int(2e5)  # Number of epochs
VERBOSE = 1  # Verbose
SAVE_BEST = False  # Save best model
WANDB_SAVE_MODEL = False  # Save model to wandb
FREEZING = [False]

# Model Architecture
MLP_HIDDENS = [512, 32, 256, 32, 128, 32, 64, 32]  # Hidden layers

PROJ_HIDDENS = [16]  # Projection hidden layers
PROJ_HIDDENS_B = [16, 8, 4, 2]  # Projection hidden layers for pretraining
EMBED_DIM = 32  # Representation dimension
DROPOUT = 0.5  # Dropout rate
ACTIVATION = None  # No activation for regression so default is LeakyReLU
NORM = 'batch_norm'  # Use batch normalization
RESIDUAL = True  # Use residual connections
SKIPPED_LAYERS = 1
SKIP_REPR = True  # residual representation

# Loss and Optimization
LOSS_KEY = 'cmse'  # Correlated Mean squared error regression loss
START_LR = 5e-4  # starting learning rate
WEIGHT_DECAY = 1  # Higher weight decay
NORMALIZED_WEIGHTS = True  # Use normalized weights

# Learning Rate Scheduling
LR_CB_MIN_LR = 1e-5  # minimum learning rate
LR_CB_FACTOR = 0.95  # factor for reducing learning rate # gradual decay leads to more stable training
LR_CB_PATIENCE = 50  # patience for reducing learning rate
LR_CB_MIN_DELTA = 1e-5 # Minimum delta for reducing learning rate
LR_CB_MONITOR = 'loss'  # Monitor validation loss

# Early Stopping
PATIENCE = int(3e3)  # Higher patience
ES_CB_MONITOR = 'val_loss'  # Monitor validation loss
ES_CB_RESTORE_WEIGHTS = True  # Restore weights

# Data Filtering and Processing
SEP_THRESHOLD = 2.30258509299  # Threshold for SEP events
BANDWIDTH = 0.88  # Bandwidth for rebalancing

# Smoothing Parameters
SMOOTHING_METHOD = 'moving_average'
WINDOW_SIZE = 61  # NOTE: must be odd
VAL_WINDOW_SIZE = 61 # NOTE: must be odd

# Additional Parameters
RHO = [0]
REWEIGHTS = [(0.85, 0.85, 0.0, 0.0)]
LAMBDA_FACTOR = 1
CVRG_MIN_DELTA = 1e-3
CVRG_MIN_DELTA_PDS = 1e-4
CVRG_METRIC = 'val_loss'
ASYM_TYPE = None #'sigmoid'
N_FILTER = 500


###############
# Pretraining related hyperparams
# PDC
EPOCHS_PRE = int(9e4)  # Higher patience for pretraining
BATCH_SIZE_PRE = 750  # Batch size for pretraining
START_LR_PRE = 1e-4 # starting learning rate for pretraining
LR_CB_MIN_LR_PRE = 1e-5  # Minimum learning rate for pretraining
LR_CB_FACTOR_PRE = 0.95 # factor for reducing learning rate in pretraining
LR_CB_PATIENCE_PRE = 50 # patience for reducing learning rate in pretraining
PATIENCE_PRE = 3300  # Higher patience for pretraining
RHO_PRE = [0.0]  # Pretraining rho parameter
WEIGHT_DECAY_PRE = 5e-3 # Higher weight decay for projection layers
WEIGHT_DECAY_PDS = 1e-6 # Higher weight decay for projection layers
WINDOW_SIZE_PRE = 11  # NOTE: must be odd
VAL_WINDOW_SIZE_PRE = 11  # NOTE: must be odd
DROPOUT_PRE = 0.4  # Dropout rate for pretraining
AE_LAMBDA = 1

# PDS configurations
START_LR_PDS = 1e-4
LR_CB_MIN_LR_PDS = 1e-6
LR_CB_FACTOR_PDS = 0.99
LR_CB_PATIENCE_PDS = 50
REWEIGHTS_PDS = [(0.2, 0.2)]  # PDS reweighting parameters
WINDOW_SIZE_PDS = 25  # NOTE: must be odd
VAL_WINDOW_SIZE_PDS = 25  # NOTE: must be odd

# PDC Weight Path
PDC_WEIGHT_PATH = "/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_pdcStratInj_bs6000_v8_20241203-194954.h5"


