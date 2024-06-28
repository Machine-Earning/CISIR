# Global configurations
SEED = [456789]
INPUTS_TO_USE = [['e0.5', 'e1.8', 'p']]
OUTPUTS_TO_USE = ['delta_p']
CME_SPEED_THRESHOLD = [0]
ALPHAS = [0.1, 0.5]
ADD_SLOPE = [False]
BATCH_SIZE = 4096
EPOCHS = int(1e6)
HIDDENS = [
    2048, 1024, 2048, 1024, 1024, 512, 1024, 512, 
    512, 256, 512, 256, 256, 128, 256, 128, 
    256, 128, 128, 128, 128, 128, 128, 128
]
PROJ_HIDDENS = [64]
LOSS_KEY = 'mse'
REPR_DIM = 128
OUTPUT_DIM = len(OUTPUTS_TO_USE)
DROPOUT = 0.1
ACTIVATION = None
NORM = 'batch_norm'
RESIDUAL = True
SKIPPED_LAYERS = 2
N = 500  # Number of samples to keep outside the threshold
LOWER_THRESHOLD = -0.5  # Lower threshold for delta_p
UPPER_THRESHOLD = 0.5  # Upper threshold for delta_p
MAE_PLUS_THRESHOLD = 0.5  # Threshold for measuring raising edges in delta
LEARNING_RATE = 3e-3  # Lower due to finetuning
WEIGHT_DECAY = 1e-6  # Higher weight decay
MOMENTUM_BETA1 = 0.9  # Higher momentum beta1
BANDWIDTH = 4.42e-2  # Bandwidth for rebalancing
PATIENCE = int(25e3)  # Higher patience
LR_CB_FACTOR = 0.9
LR_CB_PATIENCE = 1000
LR_CB_MIN_LR = 1e-4