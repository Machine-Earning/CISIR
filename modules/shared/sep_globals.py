# Dataset configurations
DS_VERSION = 1  # Dataset version
DS_PATH = 'data/sep_cme'  # Path to the dataset

OUTPUT_DIM = 1


# Training configurations
TRIAL_SEEDS = [456789, 42, 123, 0, 9999] # Seeds for trial
BATCH_SIZE = 128  # Batch size
EPOCHS = int(2e5)  # Number of epochs
VERBOSE = 1  # Verbose
SAVE_BEST = False  # Save best model
WANDB_SAVE_MODEL = False  # Save model to wandb
FREEZING = [False]

# Model Architecture
MLP_HIDDENS = [256, 64, 128, 32, 64, 32]  # Hidden layers

PROJ_HIDDENS = [32]  # Projection hidden layers
EMBED_DIM = 32  # Representation dimension
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
SEP_THRESHOLD = 2.30258509299  # Threshold for SEP events
BANDWIDTH = 7e-2 #4.42e-2  # Bandwidth for rebalancing

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

