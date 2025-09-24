# Dataset configurations
DS_VERSION = 8  # Dataset version
DS_PATH = 'data/electron_delta'  # Path to the dataset

INPUTS_TO_USE = [['e0.5', 'e4.4', 'p6.1', 'p']]  # Inputs to use
OUTPUTS_TO_USE = ['delta_p']  # Output to use
OUTPUT_DIM = len(OUTPUTS_TO_USE)  # Number of outputs
ADD_SLOPE = [False]  # Add slope to the inputs
CME_SPEED_THRESHOLD = [0]  # CME speed threshold

# Training configurations
SEEDS = [456789] # Seeds for reproducibility
TRIAL_SEEDS = [456789, 42, 123, 0, 9999] # Seeds for trial
BATCH_SIZE = 2400  # Batch size
EPOCHS = int(2e5)  # Number of epochs
VERBOSE = 1  # Verbose
SAVE_BEST = False  # Save best model
WANDB_SAVE_MODEL = False  # Save model to wandb
FREEZING = [False]


# Model Architecture
MLP_HIDDENS = [2048, 128, 1024, 128, 512, 128, 256, 128]  # Hidden layers

# MLP_HIDDENS = [2048, 1024, 512, 256, 128]  # Hidden layers
PROJ_HIDDENS = [64]  # Projection hidden layers
EMBED_DIM = 128  # Representation dimension
DROPOUT = 0.2  # Dropout rate
ACTIVATION = None  # No activation for regression so default is LeakyReLU
NORM = 'batch_norm'  # Use batch normalization
RESIDUAL = True  # Use residual connections
SKIPPED_LAYERS = 1
SKIP_REPR = True  # residual representation

# Loss and Optimization
LOSS_KEY = 'cmse'  # Correlated Mean squared error regression loss
START_LR = 1e-4  # starting learning rate
WEIGHT_DECAY = 1e-1  # Higher weight decay
MOMENTUM_BETA1 = 0.9  # Higher momentum beta1
RECIPROCAL_WEIGHTS = False  # Use reciprocal weights
NORMALIZED_WEIGHTS = True  # Use normalized weights

# Learning Rate Scheduling
LR_CB_MIN_LR = 1e-5  # minimum learning rate
LR_CB_FACTOR = 0.95  # factor for reducing learning rate # gradual decay leads to more stable training
LR_CB_PATIENCE = 50  # patience for reducing learning rate
LR_CB_MIN_DELTA = 1e-5 # Minimum delta for reducing learning rate
LR_CB_MONITOR = 'loss'  # Monitor validation loss

# Early Stopping
PATIENCE = int(4e3)  # Higher patience
ES_CB_MONITOR = 'val_loss'  # Monitor validation loss
ES_CB_RESTORE_WEIGHTS = True  # Restore weights

# Data Filtering and Processing
N_FILTERED = 500  # Number of samples to keep outside the threshold
LOWER_THRESHOLD = -0.5  # Lower threshold for delta_p
UPPER_THRESHOLD = 0.5  # Upper threshold for delta_p
MAE_PLUS_THRESHOLD = 0.5  # Threshold for measuring raising edges in delta
BANDWIDTH = 7e-2 #4.42e-2  # Bandwidth for rebalancing
TARGET_MIN_NORM_WEIGHT = 0.01  # Minimum weight for the target normalization

# Smoothing Parameters
SMOOTHING_METHOD = 'moving_average'
WINDOW_SIZE = 121  # NOTE: must be odd
VAL_WINDOW_SIZE = 121 # NOTE: must be odd

# Additional Parameters
RHO = [0]
REWEIGHTS = [(0.85, 0.85, 0.0, 0.0)]
LAMBDA_FACTOR = 1
CVRG_MIN_DELTA = 1e-3
CVRG_METRIC = 'val_loss'
ASYM_TYPE = None #'sigmoid'

LEAKY_RELU_ALPHA = 0.3


# 
FREQ_RANGE = [(-0.5, 0.5)]
MIDD_RANGE = [(-1, -0.5), (0.5, 1)]
RARE_RANGE = [(-2.5, -1), (1, 2.5)]

