from dataclasses import dataclass

'''
The code defines several data classes using the dataclass decorator
from the dataclasses module. Each data class represents a specific
configuration or set of parameters used in the model. 
'''

@dataclass
class Paths:
    rev_env: bool            # Flag to select between a reverberant (True) and non-reverberant (False) environment
    train_path: str          # Path to the training data
    test_path: str           # Path to the testing data
    results_path: str        # Path to store the results
    modelData_path: str      # Path to store model data
    log_path: str            # Path to store log files
    
@dataclass
class Params:
    win_length: int          # Window length
    R: int                   # hop_length - the length of the non-intersecting portion of window length 
    T: int                   # The length of the signal in the time domain 
    M: int                   # Number of microphones
    fs: int                  # Sample rate
    mic_ref: int             # Reference microphone index
    EnableCost: int          # Flag to enable regularization term in the loss function

@dataclass
class ModelParams:
    EnableSkipAttention: int # Flag to enable attention in the skip connections
    activationSatge1: str    # Activation fnction at the end of the UNET in stage1
    activationSatge2: str    # Activation fnction at the end of the UNET in stage2
    channelsStage1: int      # Number of input channels for stage1  
    channelsStage2: int      # Number of input channels for stage2
    numFeatures: int         # Number of features (number of frames) at the end of stage1

@dataclass
class Device:
    device_num: int          # Device number

@dataclass
class Loss:
    loss: str                 # Loss function
    norm: int                 # Normalization flag (1 or 0)

@dataclass
class Model_HP:
    train_size_spilt: float     # Training size split
    val_size_spilt: float       # Validation size split
    batchSize: int              # Batch size
    epochs: int                 # Epochs
    data_loader_shuffle: bool   # Flag to shuffle data in data loader
    test_loader_shuffle: bool   # Flag to shuffle data in test loader

@dataclass
class Optimizer:   
    optimizer: str           # Optimizer name
    learning_rate: float     # Learning rate
    weight_decay: float      # Weight decay

@dataclass 
class CUNETConfig:
    paths: Paths             # Paths configuration
    params: Params           # Parameters configuration
    modelParams: ModelParams # Model Parameters configuration
    device: Device           # Device configuration
    loss: Loss               # Loss configuration
    model_hp: Model_HP       # Model hyperparameters configuration
    optimizer: Optimizer     # Optimizer configuration
