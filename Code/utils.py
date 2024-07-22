import torch

def changeTime(signal, length):
    """
    Truncates the signal to a specified length.

    Args:
        signal (torch.Tensor): Input signal.
        length          (int): Desired length of the signal.

    Returns:
        torch.Tensor: Modified signal.
    """
    B, samples, M = signal.shape
    if samples > length:
        signal = signal[:, 0:length, :]
    signal = torch.squeeze(signal, dim=2)
    return signal

def Preprocesing(y, win_len, fs, T, R, device):
    """
    Applies preprocessing (truncates the signal and STFT) to the input signal.

    Args:
        y (torch.Tensor): Input signal in the time domain.
        win_len    (int): Window length for STFT.  
        fs         (int): Sample rate.
        T          (int): Length of the signal in the time domain.
        R          (int): hop_length - the length of the non-intersecting portion of window length.
        device (torch.device): Device to perform the operations on.

    Returns:
        Y     (torch.Tensor): Preprocessed signal in the STFT domain.
    """
    # Truncates the signal to a specified length.
    B, samples, M = y.size()
    if samples > fs * T:
        y = y[:,0:fs*T,:]                                  # y = B,T*fs,M = B,8,64000 
    y = y.permute(0,2,1).contiguous().view(B*M,-1)         # y = B*M,T*fs = B*8,64000 

    # STFT transformation 
    w_analysis = torch.hamming_window(win_len).to(device)
    Y = torch.stft(y, n_fft=win_len, hop_length=int(R), win_length=win_len, window=w_analysis, center=False)
    B_M, F, L, C = Y.size()                                # Y = B*M,F,L,C = B*8,257,497,2 
    Y = Y.permute(0,1,3,2).contiguous().view(B,M,F*C,L)    # Y = B,M,F*C,L = B,8,257*2,497

    return Y

def Postprocessing(X_hat, R, win_len, device):
    """
    Applies postprocessing (ISTFT) to the estimated source signal.

    Args:
        X_hat   (torch.Tensor): Single channel estimated source signal in the STFT domain.
        R       (int): hop_length - the length of the non-intersecting portion of window length.
        win_len (int): Window length for ISTFT.
        device  (torch.device): Device to perform the operations on.

    Returns:
        x_hat (torch.Tensor): Postprocessed signal in the time domain.
    """
    X_hat = torch.view_as_real(X_hat) # X_hat = B,F,L,2 = B,257,497,2

    # ISTFT transformation 
    w_analysis = torch.hamming_window(win_len).to(device)
    x_hat = torch.istft(X_hat, n_fft=win_len, hop_length=int(R), win_length=win_len, window=w_analysis, center=False)

    return x_hat

def beamformingOpreation(Y, mic_ref, W=0):
    """
    Applies beamforming operation to the input signal with the calculated weights.

    Args:
        Y (torch.Tensor): Input signal in the STFT domain.
        mic_ref    (int): Reference microphone index.
        W (torch.Tensor or int, optional): Beamforming weights. Defaults to 0 (uniform weights).

    Returns:
        X_hat (torch.Tensor): Estimated source signal in the STFT domain.
        Y     (torch.Tensor): Input signal in the STFT domain.
        W     (torch.Tensor or int): Updated beamforming weights. 
    """
    B, M, F, L = Y.size()        # Y = B,M,F,L = B,8,514,497
    Y = Y.view(B, M, F // 2, 2, L).permute(0,1,2,4,3).contiguous() #  Y = B,M,F//2,L,2 = B,8,257,497,2
    Y = torch.view_as_complex(Y) # Y = B,M,F//2,L = B,8,257,497

    # If we got no input weights, we will take the y recorded at the reference microphone by 
    # setting W as 1 in the reference channel and 0 for the other channels.
    if type(W) == int:  
        W = torch.zeros_like(Y)
        W[:, mic_ref-1, :, :] = W[:, mic_ref-1, :, :] + 1

    # BeamformingOpreation 
    X_hat = torch.mul(torch.conj(W), Y) # X_hat = B,M,F,L = B,8,257,497
    X_hat = torch.sum(X_hat, dim=1)     # X_hat = B,F,L = B,257,497

    return X_hat, Y, W

