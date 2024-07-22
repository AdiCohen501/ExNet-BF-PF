import torch
from utils import Preprocesing, Postprocessing, beamformingOpreation
from test import test
from ComputeLoss import Loss
import criterionFile

def train(model, args, results_path, train_loader, val_loader, optimizer, device, cfg_loss, debug):
     
    # Parameters
    fs = args.fs
    win_len = args.win_length
    T = args.T
    R = eval(args.R)
    mic_ref = args.mic_ref
    
    # Init Variables 
    epoch_train_loss, epoch_val_loss = 0 ,0
    
    model.train()
    
    for i, (y, labels_x) in enumerate(train_loader): # on batch
        # Extract Data
        y = y.to(device)                    # y = B,T*fs,M - noisy signal in the time domain
        fullLabels_x = labels_x.to(device)  # x = B,T*fs,M - target signal in the time domain  
        labels_x = torch.unsqueeze(fullLabels_x[:,:,mic_ref-1],2) # x_ref - B,T*fs,1 - target signal ref in the time domain  
        
        # Perform STFT and Padding
        Y = Preprocesing(y, win_len, fs, T, R, device)   # Y = B,M,2*F,L - noisy signal in the STFT domain
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        _,X_hat_Stage1,Y,W_Stage1,X_hat_Stage2,_,_,_ = model(Y, device)

        # Perform ISTFT and norm for x_hat before PF
        x_hat_stage1_B_norm = Postprocessing(X_hat_Stage1,R,win_len,device)
        max_x = torch.max(abs(x_hat_stage1_B_norm),dim=1).values
        x_hat_stage1 = (x_hat_stage1_B_norm.T/max_x).T
        
        # Perform ISTFT and norm for x_hat
        x_hat_stage2_B_norm = Postprocessing(X_hat_Stage2,R,win_len,device)
        max_x = torch.max(abs(x_hat_stage2_B_norm),dim=1).values
        x_hat_stage2= (x_hat_stage2_B_norm.T/max_x).T       
        
        # Preprocessing & Postprocessing for the labeled signal
        X_stft = Preprocesing(fullLabels_x, win_len, fs, T, R, device)
        X_stft_mic_ref,_,_ =  beamformingOpreation(X_stft,mic_ref)
        x = Postprocessing(X_stft_mic_ref,R,win_len,device)
        max_x = torch.max(abs(x),dim=1).values
        x = (x.T/max_x).T

        # Calculate the loss function 
        loss = Loss(x,x_hat_stage2,cfg_loss)

        # Calculate the cost function (regularization term)
        if args.EnableCost:
            # Beamforming opreation on x (calculate ISTFT(conj(W)X))
            WX,_,_ =  beamformingOpreation(X_stft,mic_ref,W_Stage1)
            wx = Postprocessing(WX,R,win_len,device)
            max_x = torch.max(abs(wx),dim=1).values
            wx = (wx.T/max_x).T
            # Calculate MAE |x-w*x|
            criterion_L1 = criterionFile.criterionL1 
            cost_wx = criterion_L1(wx.float(), x.float(),cfg_loss.norm) 
            cost_wx = ((cost_wx)*10000)
            loss = loss + cost_wx   

        # Backward & Optimize
        loss.backward()
        if debug:
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name, param.grad)
        optimizer.step()        
        epoch_train_loss += loss.item() 

    # VAL    
    epoch_val_loss = test(model, args, results_path, val_loader, device, cfg_loss, 0)
   
    return epoch_train_loss,epoch_val_loss