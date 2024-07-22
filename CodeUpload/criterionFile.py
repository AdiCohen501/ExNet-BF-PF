import torch
import torch.nn as nn

def criterionL1(outputs, labels, enable_alpha): 
    """
    Calculates the L1 loss between the predicted outputs and the target labels.
    
    Args:
        outputs (torch.Tensor): Predicted outputs from the model.
        labels  (torch.Tensor): Target labels.
        enable_alpha    (bool): Flag to enable alpha calculation (alpha = dot(x_hat,x)/dot(x,x)).
    
    Returns:
        torch.Tensor: Computed L1 loss.
    """
    if enable_alpha:
        alpha = (torch.mul(outputs, labels)).sum(1) / (torch.mul(labels, labels)).sum(1)
        alpha = alpha.unsqueeze(dim=1)
        labels = torch.mul(alpha, labels)
    criterionL1 = nn.L1Loss()
    loss = criterionL1(outputs, labels)
    return loss

