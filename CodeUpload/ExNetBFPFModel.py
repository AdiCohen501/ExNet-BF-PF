import torch
import torch.nn as nn
from UnetModel import UNET
from Beamformer import beamformingOperationStage1, MaskOperationStage2

class ExNetBFPF(nn.Module):
    """
    ExNet-BF+PF Model
    """
    def __init__(self,modelParams):
        """
        Args:
            modelParams -  Model Parameters configuration, contains:
                activationSatge1    (str): The activation function at the end of stage1.
                activationSatge2    (str): The activation function at the end of stage2.
                channelsStage1      (int): Number of input channels for stage1.
                channelsStage2      (int): Number of input channels for stage2.
                numFeatures         (int): Number of features (number of frames) at the end of stage1.
                EnableSkipAttention (int): Flag to enable attention in the skip connections

        Returns:
            W_timeChange   (torch.Tensor): The estimeted weights that change in time at the end of stage1.
            X_hat_Stage1_C (torch.Tensor): The estimeted signal at the end of the stage1 in the STFT domain.
            Y              (torch.Tensor): The noisy (input) signal in the STFT domain.
            W_Stage1       (torch.Tensor): The estimated weights at the end of stage1.
            X_hat_Stage2_C (torch.Tensor): The estimated signal at the end of the stage2 in the STFT domain.
            W_Stage2       (torch.Tensor): The estimated weights at the end of stage2.
            skip_Stage1    (torch.Tensor): The attention mask of the final layer of stage1.
            skip_Stage2    (torch.Tensor): The attention mask of the final layer of stage2.
        """
        super(ExNetBFPF, self).__init__()
        # Create the first and second stage UNET
        self.unet_multiChannel  = UNET(modelParams.channelsStage1,modelParams.activationSatge1,
                                       modelParams.EnableSkipAttention)
        self.unet_singleChannel = UNET(modelParams.channelsStage2,modelParams.activationSatge2,
                                       modelParams.EnableSkipAttention)
    
    def forward(self, Y, device):   
        # Stage 1: Multi-Channel speech enhancement
        W_timeChange,skip_Stage1 = self.unet_multiChannel(Y.float()) # W changes over time

        # Apply mean to W_timeChange for getting timeFxed weights 
        W_timeFixed = torch.mean(W_timeChange,3)  
        
        # Force frequencies 0 and pi to be real 
        W_timeFixed[:,:,1]  = torch.zeros_like(W_timeFixed[:,:,1])
        W_timeFixed[:,:,-1] = torch.zeros_like(W_timeFixed[:,:,-1])
        
        # Beamforming Operation 
        X_hat_Stage1_C,X_hat_satge1,Y,W_Stage1 = beamformingOperationStage1(Y, W_timeFixed)     

        # Stage 2: Single-Channel speech enhancement
        W_Stage2,skip_Stage2 = self.unet_singleChannel(X_hat_satge1.float())

        # Mask Operation
        X_hat_Stage2_C,_,_ = MaskOperationStage2(X_hat_Stage1_C, W_Stage2)       
        
        return W_timeChange,X_hat_Stage1_C,Y,W_Stage1,X_hat_Stage2_C,W_Stage2,skip_Stage1,skip_Stage2