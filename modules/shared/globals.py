# Global configurations
SEEDS = [456789, 42, 1234]  # Seeds for reproducibility
INPUTS_TO_USE = [['e0.5', 'e4.4', 'p6.1', 'p']]  # Inputs to use
OUTPUTS_TO_USE = ['delta_p']  # Output to use
CME_SPEED_THRESHOLD = [0]  # CME speed threshold
# ALPHAS = [0, 0.3, 0.2, 0.1, 0.4]  # Alpha values for the loss
ALPHAS = [1.0, 0.7, 0.5]  # Alpha values for the loss
ADD_SLOPE = [False]  # Add slope to the inputs
BATCH_SIZE = 4096  # Batch size
EPOCHS = int(1e6)  # Number of epochs
MLP_HIDDENS = [
    2048, 1024, 2048, 1024, 1024, 512, 1024, 512,
    512, 256, 512, 256, 256, 128, 256, 128,
    256, 128, 128, 128, 128, 128, 128, 128
]  # Hidden layers
PROJ_HIDDENS = [64]  # Projection hidden layers
LOSS_KEY = 'mse_pcc'  # Mean squared error regression loss
LAMBDA = 3.3  # Lambda for the loss
REPR_DIM = 128  # Representation dimension
OUTPUT_DIM = len(OUTPUTS_TO_USE)  # Number of outputs
DROPOUT = 0.1  # Dropout rate
ACTIVATION = None  # No activation for regression
NORM = 'batch_norm'  # Use batch normalization
RESIDUAL = True  # Use residual connections
SKIPPED_LAYERS = 2  # Number of layers to skip in residual connections
N_FILTERED = 500  # Number of samples to keep outside the threshold
LOWER_THRESHOLD = -0.5  # Lower threshold for delta_p
UPPER_THRESHOLD = 0.5  # Upper threshold for delta_p
MAE_PLUS_THRESHOLD = 0.5  # Threshold for measuring raising edges in delta
START_LR_FT = 3e-3  # Lower due to fine-tuning
START_LR = 1e-2  # starting learning rate
START_LR_PDS = 7e-2  # starting learning rate
WEIGHT_DECAY = 1e-5  # Higher weight decay
WEIGHT_DECAY_PDS = 1e-5  # Higher weight decay for projection layers
MOMENTUM_BETA1 = 0.9  # Higher momentum beta1
BANDWIDTH = 4.42e-2  # Bandwidth for rebalancing
PATIENCE = int(3e3)  # Higher patience
LR_CB_FACTOR = 0.9  # factor for reducing learning rate
LR_CB_PATIENCE = 500  # patience for reducing learning rate
LR_CB_MIN_LR = 1e-6  # minimum learning rate
VERBOSE = 1  # Verbose
SAVE_BEST = False  # Save best model
LR_CB_MIN_DELTA = 1e-5  # Minimum delta for reducing learning rate
LR_CB_MONITOR = 'loss'  # Monitor validation loss
DS_VERSION = 7  # Dataset version
DS_PATH = 'data/electron_cme_data_split_v7'  # Path to the dataset
VAL_SPLIT = 0.25  # Validation split
TARGET_MIN_NORM_WEIGHT = 0.01  # Minimum weight for the target normalization
COMMON_VAL_ALPHA = 0.75  #1  # Common validation alpha
ES_CB_MONITOR = 'val_loss'  # Monitor validation loss
ES_CB_RESTORE_WEIGHTS = True  # Restore weights
WANDB_SAVE_MODEL = False  # Save model to wandb
RECIPROCAL_WEIGHTS = False  # Use reciprocal weights
SAM_RHOS = [0.05, 0.5]  # Rho values for SAM

# ATTM AREA
BLOCKS_HIDDENS = [128 for _ in range(24)]
ATTN_HIDDENS = [128 for _ in range(12)]
ATTM_START_LR = 1e-5
ATTM_ACTIVATION = 'leaky_relu'
ATTN_SKIPPED_LAYERS = 1
ATTM_SKIPPED_BLOCKS = 1
ATTN_RESIDUAL = False
ATTM_RESIDUAL = True
ATTM_DROPOUT = 0
ATTN_DROPOUT = 0
ATTN_NORM = None
ATTM_NORM = 'batch_norm'
ATTM_LR_CB_MIN_LR = 1e-6
ATTM_WD = 1e-9
ATTM_LR_CB_FACTOR = 0.9
FOLD_PATIENCE = int(3e3)
