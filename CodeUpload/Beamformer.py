import torch
    
def beamformingOperationStage1(Y, W):
    """
    Beamformring Operation
    Args:
        Y (torch.Tensor): The noisy (input) signal in the STFT domain.
        W (torch.Tensor): The estimated weights at the end of stage1.

    Returns:
        X_hat_complex (torch.Tensor): The estimated signal at the end of the stage1 in complex form.
        X_hat         (torch.Tensor): The estimated signal at the end of the stage1.
        Y             (torch.Tensor): The noisy (input) signal in complex form.
        W             (torch.Tensor): The estimated weights at the end of stage1 in complex form.
    """
    
    # Change sizes and form 
    B, M, F, L = Y.size()                                       # Y = B,M,F,L = B,8,514,497
    Y = Y.view(B,M,F//2,2,L).permute(0,1,2,4,3).contiguous()    # Y = B,M,F//2,L,2 = B,8,257,497,2
    B, M, F = W.size()                                          # W = B,M,F = B,8,514
    W = W.view(B,M,F//2,2).contiguous()                         # W = B,M,F//2,2 = B,8,257,2

    Y = torch.view_as_complex(Y)                                # Y = B,M,F//2,L = B,8,257,497
    W = torch.view_as_complex(W)                                # W = B,M,F//2   = B,8,257
    W = W.unsqueeze(dim=3)                                      # W = B,M,F//2,1 = B,8,257,1
    
    # Beamforming Opreation
    X_hat = torch.mul(torch.conj(W), Y)                         # X_hat = B,M,F,L = B,8,257,497
    X_hat_complex = torch.sum(X_hat,dim = 1)                    # X_hat = B,F,L   = B,257,497
    X_hat = torch.view_as_real(X_hat_complex)                   # X_hat = B,F,L,C = B,257,497,2
    
    # Change sizes and form 
    B, F, L, C = X_hat.size()
    M = 1
    X_hat = X_hat.permute(0,1,3,2).contiguous().view(B,M,F*C,L) # X_hat = B,M,F*C,L = B,1,514,497

    return X_hat_complex,X_hat,Y,W


def MaskOperationStage2(Y, W):
    """
    Mask Operation
    Args:
        Y (torch.Tensor): The estimated signal at the end of the stage1.
        W (torch.Tensor): The estimated weights at the end of stage2.

    Returns:
        X_hat_complex (torch.Tensor): The estimated signal at the end of the stage2 in complex form.
        Y             (torch.Tensor): The estimated signal at the end of the stage1 in complex form.
        W             (torch.Tensor): The estimated weights at the end of stage2 in complex form.
    """
    
    # Change sizes and form 
    B, M, F, L = W.size()                                    # W = B,M,F,L = B,1,514,497
    W = W.view(B,M,F//2,2,L).permute(0,1,2,4,3).contiguous() # W = B,M,F//2,L,2 = B,1,257,497,2

    # Force frequencies 0 and pi to be real 
    W[:,:,0,:,1] = torch.zeros_like(W[:,:,0,:,1])
    W[:,:,-1,:,1] = torch.zeros_like(W[:,:,-1,:,1])
    
    W = torch.view_as_complex(W).squeeze()                   # W = B,F//2,L = B,257,497
    
    # Mask Operation
    X_hat_complex = torch.mul(torch.conj(W), Y)              # X_hat_complex = B,F,L = B,257,497

    return X_hat_complex,Y,W
