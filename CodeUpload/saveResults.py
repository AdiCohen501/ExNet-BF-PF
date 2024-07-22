import scipy.io as sio
from datetime import datetime
import os

def saveResults(Y,X,skip_Stage1,skip_Stage2,W,W_timeChange,W_Stage2,X_hat_Stage1,X_hat_Stage2,
                y,x_hat_stage1,x_hat_stage2,results_path,i,fs):
          
    # Create the directory to save results if it doesn't exist
    os.makedirs(results_path, mode=0o777, exist_ok=True)
    
    # Get the current timestamp
    now = (datetime.now()).strftime("%d_%m_%Y__%H_%M_%S")

    ####### Save STFT Domain #######
    # Save Y
    Y_STFT = (Y).cpu()                       # Convert tensor to CPU
    Y_STFT = (Y_STFT).detach().numpy()       # Convert tensor to numpy array
    # Save X 
    X_STFT = (X).cpu()
    X_STFT = (X_STFT).detach().numpy()
    # Save skip connection of stage 1 
    skip_Stage1 = (skip_Stage1).cpu()
    skip_Stage1 = (skip_Stage1).detach().numpy()
    # save skip connection of stage 2
    skip_Stage2 = (skip_Stage2).cpu()
    skip_Stage2 = (skip_Stage2).detach().numpy()
    # Save W timeFixed 
    W_STFT_timeFixed = (W).cpu()
    W_STFT_timeFixed = (W_STFT_timeFixed).detach().numpy()
    # Save W timeChange 
    W_STFT_timeChange = (W_timeChange).cpu()
    W_STFT_timeChange = (W_STFT_timeChange).detach().numpy()
    # save W of stage 2
    W_Stage2 = (W_Stage2).cpu()
    W_Stage2 = (W_Stage2).detach().numpy()
    # save X_hat stage1 
    X_hat_Stage1 = (X_hat_Stage1).cpu()
    X_hat_Stage1 = (X_hat_Stage1).detach().numpy()
    # save X_hat stage2
    X_hat_Stage2 = (X_hat_Stage2).cpu()
    X_hat_Stage2 = (X_hat_Stage2).detach().numpy()

    # Define the file path and name for STFT domain results
    filename = results_path + 'TEST_STFT_domain_results_' + now + '_' + str(i) + '.mat'

    # Create a dictionary with the variable names and their corresponding values
    mdic = {
        "Y_STFT": Y_STFT,
        "X_STFT": X_STFT,
        "W_STFT_timeChange": W_STFT_timeChange,
        "W_STFT_timeFixed": W_STFT_timeFixed,
        "W_Stage2": W_Stage2,
        "X_hat_Stage1": X_hat_Stage1,
        "X_hat_Stage2": X_hat_Stage2,
        "skip_Stage1": skip_Stage1,
        "skip_Stage2": skip_Stage2
    }

    # Save the dictionary as .mat file
    sio.savemat(filename, mdic)
    #################################

    ####### Save Time Domain #######
    # Save y
    y = y.cpu()   # Convert tensor to CPU
    # Save x_hat stage1
    x_hat_stage1 = x_hat_stage1.cpu()
    # Save x_hat stage2
    x_hat_stage2 = x_hat_stage2.cpu()

    # Define the file path and name for time domain results
    filename = results_path + 'TEST_time_domain_results_' + now + '_' + str(i) + '.mat'

    # Create a dictionary with the variable names and their corresponding values
    mdic = {
        "y": y.numpy(),
        "x_hat_stage1": x_hat_stage1.numpy(),
        "x_hat_stage2": x_hat_stage2.numpy()
    }
    
    # Save the dictionary as .mat file
    sio.savemat(filename, mdic)

