import torch
from ExNetBFPFModel import ExNetBFPF

def loadPreTrainedModel(cfg):
       
    # Defining the model
    model = ExNetBFPF(cfg.modelParams)

    if cfg.paths.rev_env:  # Reverberant Environment
        PATH = cfg.paths.modelData_path + 'trained_model_dataset_withRev_two_step_approach.pt'
        # Load the model with the best saved weights
        model = torch.load(PATH)

    else: # Non-Reverberant Environment
        PATH = cfg.paths.modelData_path + 'trained_model_dataset_withoutRev.pt'
        # Load the model with the best saved weights
        model = torch.load(PATH)

    return model

