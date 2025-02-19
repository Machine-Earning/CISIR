# Dataset configurations
DS_VERSION = 8  # Dataset version
DS_PATH = 'data/electron_cme_data_split_v8'  # Path to the dataset

INPUTS_TO_USE = [['e0.5', 'e4.4', 'p6.1', 'p']]  # Inputs to use
OUTPUTS_TO_USE = ['delta_p']  # Output to use
OUTPUT_DIM = len(OUTPUTS_TO_USE)  # Number of outputs
ADD_SLOPE = [False]  # Add slope to the inputs
CME_SPEED_THRESHOLD = [0]  # CME speed threshold

# Training configurations
SEEDS = [456789]  # , 42, 1234, 0, 9999]  # Seeds for reproducibility
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
START_LR = 3e-4  # starting learning rate
WEIGHT_DECAY = 1e-4  # Higher weight decay
MOMENTUM_BETA1 = 0.9  # Higher momentum beta1
RECIPROCAL_WEIGHTS = False  # Use reciprocal weights

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
N_FILTERED = 500  # Number of samples to keep outside the threshold
LOWER_THRESHOLD = -0.5  # Lower threshold for delta_p
UPPER_THRESHOLD = 0.5  # Upper threshold for delta_p
MAE_PLUS_THRESHOLD = 0.5  # Threshold for measuring raising edges in delta
BANDWIDTH = 7e-2 #4.42e-2  # Bandwidth for rebalancing
TARGET_MIN_NORM_WEIGHT = 0.01  # Minimum weight for the target normalization

# Smoothing Parameters
SMOOTHING_METHOD = 'moving_average'
VAL_WINDOW_SIZE = 101 #5  # NOTE: must be odd
WINDOW_SIZE = 101 #121  # NOTE: must be odd

# Additional Parameters
RHO = [1e-2]
REWEIGHTS = [(1.0, 0.4, 0.1, 0)]
LAMBDA_FACTOR = 8
AE_LAMBDA = 0.9
CVRG_MIN_DELTA = 1e-3
CVRG_METRIC = 'loss'
CVRG_METRIC_WDR = 'val_loss'
ASYM_TYPE = None #'sigmoid'

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
ATTM_RHO = [1e-3]  #[1e-3]
ATTM_PATIENCE = int(2e3)
ATTM_CVRG_MIN_DELTA = 1e-2
ATTM_VAL_WINDOW_SIZE = 15
ATTM_WINDOW_SIZE = 15
LAMBDA_FACTOR_ATTM = 6

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
COMBINER_OUTPUT_DIM = 3  # 3 classes for routing
BATCH_SIZE_MOE = 20  # Batch size for Moe
BATCH_SIZE_MOE_0 = 2048  # Batch size for Moe
PLUS_INDEX = 0
MID_INDEX = 1
MINUS_INDEX = 2
RHO_MOE_C = [1e-2]
RHO_MOE_0 = [1e-1]
RHO_MOE_P = [5e-1]
RHO_MOE_M = [5e-1]
PATIENCE_MOE_C = int(2e3)
PATIENCE_MOE_M = int(3.3e3)
PATIENCE_MOE_P = int(3.3e3)
PATIENCE_MOE_0 = int(3.3e3)

LOWER_THRESHOLD_MOE = -0.4
UPPER_THRESHOLD_MOE = 0.4
REWEIGHTS_MOE_C = [(0.65, 0.65, 0.0, 0.0)]
REWEIGHTS_MOE_P = [(0.11, 0.11, 0.0, 0.0)]
REWEIGHTS_MOE_M = [(0.03, 0.03, 0.0, 0.0)]
REWEIGHTS_MOE_0 = [(0.4, 0.4, 0.0, 0.0)]  # [(0.0, 0.0, 0.0, 0.0)]
LAMBDA_FACTOR_MOE_P = 6
LAMBDA_FACTOR_MOE_M = 6

START_LR_MOE_M = 1e-4
START_LR_MOE_P = 1e-4

FOP_FACTOR = 1.5  # 1 full penalty, 0 no penalty

ASYM_TYPE_0 = None
PDC_WEIGHT_PATH = "/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_pdcStratInj_bs6000_v8_20241203-194954.h5"

PRE_WEIGHT_PATH = "/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse1.00_v8_updated_20241120-180201_reg.h5"
# Model paths
POS_EXPERT_PATH = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse0.10_plus_e_20241212-140850_reg.h5'
NEG_EXPERT_PATH = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse0.03_minus_e_20241212-133248_reg.h5'
NZ_EXPERT_PATH = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_amse0.10_zero_e_20241205-111054_reg.h5'
COMBINER_PATH = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2_ace0.65_combiner_lpn1.00_lnz1.00_dualsig_20250107-152210.h5'
COMBINER_PATH_NOC = '/home1/jmoukpe2016/keras-functional-api/final_router_model_weights_mlp2_ace0.60_router_20241218-021242.h5'
COMBINER_PDCAE_S1 = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2ae_pdcStratInj_bs3600_rho0.10_20241115-021423.h5'
COMBINER_PDCAE_S2 = '/home1/jmoukpe2016/keras-functional-api/final_model_weights_mlp2_pdcaeS2_amse1.00_v8_frFalse_20241121-102817_s2min_reg.h5'
COMBINER_V3 = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2_amse0.50_moe_cheat_v3_randInitCombiner_20250117-155615.h5'
COMBINER_PCC_CE = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2_combiner_lce0.50_20250117-133726.h5'
COMBINER_PCC_CE_PDCAE_S1 = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2pdcaes1_ace0.65_combiner_lpn0.00_lnz0.00_lce1.00_20250118-101801.h5'
COMBINER_PCC_CE_PDCAE_S2 = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2pdcaes2_combiner_lce0.50_20250118-145316.h5'
COMBINER_PCC_CE_S2 = '/home1/jmoukpe2016/keras-functional-api/final_combiner_weights_mlp2_amse0.40_moe_cheat_pcc_ce_20250117-160500.h5'
COMBINER_PN_NZ = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2_combiner_lpn1.00_lnz1.00_lce0.00_20250117-133720.h5'
COMBINER_V2_PDCAE_S1 = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2pdcaes1_ace0.65_combiner_lpn0.00_lnz0.00_lce1.00_20250118-101801.h5'
COMBINER_V2_PDCAE_S2 = '/home1/jmoukpe2016/keras-functional-api/final_combiner_model_weights_mlp2pdcaes2_ace0.65_combiner_lpn0.00_lnz0.00_lce1.00_20250118-101801.h5'
COMBINER_V3_PDCAE_S2 = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2pdcaes1_amse0.39_moe_cheat_v3_randInitCombiner_20250122-160802.h5'
COMBINER_V3_A0 = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2_amse0.00_moe_cheat_v3_randInitCombiner_20250121-142443.h5'
COMBINER_V3_NRELU = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2_amse0.10_moe_cheat_v3nrelu_randInitCombiner_20250122-162810.h5'
COMBINER_V3_PDCAE_S2_NRELU = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2pdcaes1_amse0.40_moe_cheat_v3nrelu_randInitCombiner_20250123-133559.h5'
COMBINER_V2_PCC_CE_S2_A0 = '/home1/jmoukpe2016/keras-functional-api/final_combiner_weights_mlp2_amse0.00_moe_cheat_pcc_ce_20250123-144353.h5'
COMBINER_V3_COS = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2_amse1.00_moe_cheat_v3cos_randInitCombiner_20250128-142530.h5'
# weights for investigation
COMBINER_V2_PCC_CE_S2_A0_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.00_v2_moe_cheat_pcc_ce_investigation_A_20250127-113654.h5'
COMBINER_V2_PCC_CE_S2_B0_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.00_v2_moe_cheat_pcc_ce_investigation_B_20250127-114150.h5'
COMBINER_V2_PCC_CE_S2_C0_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.00_v2_moe_cheat_pcc_ce_investigation_C_20250127-114528.h5'
COMBINER_V2_PCC_CE_S2_D0_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/final_combiner_weights_mlp2_amse0.00_v2_moe_cheat_pcc_ce_investigation_20250128-142235.h5'
COMBINER_V2_PCC_CE_S2_A04_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_A_20250127-113651.h5'
COMBINER_V2_PCC_CE_S2_B04_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_B_20250127-114151.h5'
COMBINER_V2_PCC_CE_S2_C04_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_C_20250127-114523.h5'
COMBINER_V2_PCC_CE_S2_D04_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/final_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_20250128-142357.h5'

COMBINER_V2_PCC_CE_S2_A04_INVESTIGATION_BS64 = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_A_20250203-115542.h5'
COMBINER_V2_PCC_CE_S2_A04_INVESTIGATION_BS82 = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_A_20250204-133928.h5'

COMBINER_V2_PCC_CE_S2_B04_INVESTIGATION_BS1800 = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_B_20250206-122228.h5'
COMBINER_V2_PCC_CE_S2_C04_INVESTIGATION_BS800 = '/home1/jmoukpe2016/keras-functional-api/inv_combiner_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_C_20250206-133229.h5'

MOE_V2_PCC_CE_S2_A04_INVESTIGATION = '/home1/jmoukpe2016/keras-functional-api/inv_model_moe_weights_mlp2_amse0.40_v2_moe_cheat_pcc_ce_investigation_A_20250127-113651_reg.h5'
MOE_V2_PCC_CE_S2_BS1024 = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2_amse0.40_moe_cheat_v3nrelu_20250210-145340.h5'

COMBINER_V3_AE_OF = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2pdcaes1_amse0.40_moe_cheat_v3nrelu_of_20250213-131806.h5'
COMBINER_V3_OF = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2_amse0.40_moe_cheat_v3nrelu_of_20250213-133238.h5'

COMBINER_V3_AE_NOF_1 = '/home1/jmoukpe2016/keras-functional-api/combiner_v3_weights_mlp2pdcaes1_amse0.40_moe_cheat_v3nrelu_of_20250218-132139.h5'
# assemble hyperparams
REWEIGHTS_MOE = [(0.4, 0.4, 0.0, 0.0)]
RHO_MOE = [1e-2]
LAMBDA_FACTOR_MOE = 8
START_LR_MOE = 1e-4
LR_CB_MIN_LR_MOE = 2e-5
LR_CB_FACTOR_MOE = 0.95
LR_CB_PATIENCE_MOE = 50
WEIGHT_DECAY_MOE = 1e-4
PATIENCE_MOE = int(3e3)
ASYM_TYPE_MOE = None
BATCH_SIZE_MOE_S2 = 2400

FREEZE_EXPERT = True
PRETRAINING_MOE = False
MODE_MOE = 'soft'


LAMBDA_PN_CCE = 1
LAMBDA_NZ_CCE = 1
LAMBDA_CE = 0.5

START_LR_MOE_C = 1e-4
REWEIGHTS_MOE_C2 = [(1.0, 1.0, 1.0, 1.0)]
BATCH_SIZE_NEG = 70


# INVESTIGATION
START_LR_MOE_INV = 1e-4
IMPORTANCE_MOE_INV = [(0.4, 0.4, 0.0, 0.0)]
RHO_MOE_INV = [1e-2]
LAMBDA_FACTOR_MOE_INV = 8
PATIENCE_MOE_INV = int(3e3)
ASYM_TYPE_MOE_INV = None
BATCH_SIZE_MOE_INV = 2800


# QTC
REWEIGHTS_MOE_QTC = [(0.2, 0.2, 0.0, 0.0)] # [(0.25, 0.25, 0.0, 0.0)]
