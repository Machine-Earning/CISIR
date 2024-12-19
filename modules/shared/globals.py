# Dataset configurations
DS_VERSION = 8  # Dataset version
DS_PATH = 'data/electron_cme_data_split_v8'  # Path to the dataset
INPUTS_TO_USE = [['e0.5', 'e4.4', 'p6.1', 'p']]  # Inputs to use
OUTPUTS_TO_USE = ['delta_p']  # Output to use
OUTPUT_DIM = len(OUTPUTS_TO_USE)  # Number of outputs
ADD_SLOPE = [False]  # Add slope to the inputs
CME_SPEED_THRESHOLD = [0]  # CME speed threshold

# Training configurations
SEEDS = [456789] # , 42, 1234, 0, 9999]  # Seeds for reproducibility
BATCH_SIZE = 2500  # Batch size
EPOCHS = int(1e5)  # Number of epochs
VERBOSE = 1  # Verbose
SAVE_BEST = False  # Save best model
WANDB_SAVE_MODEL = False  # Save model to wandb

# Pretraining configurations
BATCH_SIZE_PRE = 7000  # Batch size for PDS
START_LR_PRE = 6e-4  # starting learning rate for pretraining
LR_CB_MIN_LR_PRE = 6e-8  # Minimum learning rate for pretraining
LR_CB_FACTOR_PRE = 0.99  # factor for reducing learning rate in pretraining
LR_CB_PATIENCE_PRE = 50  # patience for reducing learning rate in pretraining
PATIENCE_PRE = int(1e4)  # Higher patience for pretraining
RHO_PRE = [0.2]  # Pretraining rho parameter
REWEIGHTS_PRE = [(1.0, 0.4)]  # Pretraining reweighting parameters
WEIGHT_DECAY_PRE = 1e-4  # Higher weight decay for projection layers
WINDOW_SIZE_PDC = 151  # NOTE: must be odd
VAL_WINDOW_SIZE_PDC = 151  # NOTE: must be odd
DROPOUT_PRE = 1e-2  # Dropout rate for pretraining

# PDS 
START_LR_PDS = 1e-3
LR_CB_MIN_LR_PDS = 1e-6
LR_CB_FACTOR_PDS = 0.99
LR_CB_PATIENCE_PDS = 50
REWEIGHTS_PDS = [(0.2, 0.2)]  # Pretraining reweighting parameters
WINDOW_SIZE_PDS = 25  # NOTE: must be odd
VAL_WINDOW_SIZE_PDS = 25  # NOTE: must be odd

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
START_LR = 1e-3  # starting learning rate
WEIGHT_DECAY = 1e-4  # Higher weight decay
MOMENTUM_BETA1 = 0.9  # Higher momentum beta1
RECIPROCAL_WEIGHTS = False  # Use reciprocal weights

# Learning Rate Scheduling
LR_CB_MIN_LR = 1e-5  # minimum learning rate
LR_CB_FACTOR = 0.95 # factor for reducing learning rate # gradual decay leads to more stable training
LR_CB_PATIENCE = 50  # patience for reducing learning rate
LR_CB_MIN_DELTA = 1e-5  # Minimum delta for reducing learning rate
LR_CB_MONITOR = 'loss'  # Monitor validation loss

# Early Stopping
PATIENCE = int(3e3)  # Higher patience
ES_CB_MONITOR = 'val_loss'  # Monitor validation loss
ES_CB_RESTORE_WEIGHTS = True  # Restore weights

# Data Filtering and Processing
N_FILTERED = 500  # Number of samples to keep outside the threshold
LOWER_THRESHOLD = -0.5  # Lower threshold for delta_p
UPPER_THRESHOLD = 0.5  # Upper threshold for delta_p
MAE_PLUS_THRESHOLD = 0.5  # Threshold for measuring raising edges in delta
BANDWIDTH = 4.42e-2  # Bandwidth for rebalancing
TARGET_MIN_NORM_WEIGHT = 0.01  # Minimum weight for the target normalization

# Smoothing Parameters
SMOOTHING_METHOD = 'moving_average'
VAL_WINDOW_SIZE = 101  # NOTE: must be odd
WINDOW_SIZE = 101  # NOTE: must be odd

# Additional Parameters
RHO = [1e-2]
REWEIGHTS = [(1.0, 0.3, 0.1, 0)]
LAMBDA_FACTOR = 8
AE_LAMBDA = 0.9
CVRG_MIN_DELTA = 1e-5
CVRG_METRIC = 'loss'
CVRG_METRIC_WDR = 'val_loss'
ASYM_TYPE = 'sigmoid'

# ATTM AREA
BLOCKS_HIDDENS = [128 for _ in range(3)]
ATTM_START_LR = 1e-3
ATTM_LR_CB_MIN_LR = 5e-6
ATTM_ACTIVATION = 'leaky_relu'
ATTM_SKIPPED_BLOCKS = 1
ATTM_RESIDUAL = True
ATTM_DROPOUT = 0.02
ATTM_NORM = 'batch_norm'
ATTM_WD = 1e-6
ATTM_LR_CB_FACTOR = 0.95
ATTM_LR_CB_PATIENCE = 50
ATTM_RHO = [1e-5] #[1e-3]
ATTM_PATIENCE = int(3e3)
ATTM_CVRG_MIN_DELTA = 1e-2
ATTM_VAL_WINDOW_SIZE = 33
ATTM_WINDOW_SIZE = 33
LAMBDA_FACTOR_ATTM = 4


# ATTN AREA
ATTN_HIDDENS = [256, 128, 256]  # this architecture is good enough to predict on its own
ATTN_SKIPPED_LAYERS = 1
ATTN_RESIDUAL = True
ATTN_DROPOUT = 0.2
ATTN_NORM = 'batch_norm'

# FF AREA
FF_HIDDENS = [128, 256, 128]  # this architecture is good enough to predict on its own
FF_SKIPPED_LAYERS = 1
FF_RESIDUAL = True
FF_DROPOUT = 0.2
FF_NORM = 'batch_norm'

LEAKY_RELU_ALPHA = 0.3

# MOE
ROUTER_OUTPUT_DIM = 3  # 3 classes for routing
BATCH_SIZE_MOE = 32  # Batch size for Moe
BATCH_SIZE_MOE_0 = 2048  # Batch size for Moe
PLUS_INDEX = 0
MID_INDEX = 1
MINUS_INDEX = 2
RHO_MOE_R = [1e-2] 
RHO_MOE_0 = [1e-1] 
RHO_MOE_P = [5e-1] 
RHO_MOE_M = [5e-1] 
PATIENCE_MOE = int(5e3)
PATIENCE_MOE_M = int(7e3)
PATIENCE_MOE_P = int(7e3)
PATIENCE_MOE_0 = int(7e3)

LOWER_THRESHOLD_MOE = -0.4
UPPER_THRESHOLD_MOE = 0.4
REWEIGHTS_MOE_R = [(0.69, 0.69)]  
REWEIGHTS_MOE_P = [(0.11, 0.11, 0.0, 0.0)]
REWEIGHTS_MOE_M = [(0.035, 0.035, 0.0, 0.0)]
REWEIGHTS_MOE_0 = [(0.4, 0.4, 0.0, 0.0)]  # [(0.0, 0.0, 0.0, 0.0)]
LAMBDA_FACTOR_MOE_P = 6
LAMBDA_FACTOR_MOE_M = 6
ASYM_TYPE_0 = None
ASYM_TYPE_MOE = None
PDC_WEIGHT_PATH = "/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_pdcStratInj_bs6000_v8_20241203-194954.h5"
PRE_WEIGHT_PATH = "/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse1.00_v8_updated_20241120-180201_reg.h5"
LAMBDA_1_CCE = 1.0
LAMBDA_2_CCE = 1.0
K_CCE = 5.0

