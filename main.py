from model import *
from data_generator import * 
from kshot_testing import * 
from trainer import * 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def train_full_model(task_labels_present, basic_maml, num_epochs, sinusoid=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if sinusoid == True:  # set dims for sinusoid
        input_dim = 1
        task_label_dim = 2
    else: # set dims for 2d gaussian 
        input_dim = 2
        task_label_dim = 3
        
    # Initialize model and dataset
    if basic_maml == True: 
        model = Regressor(input_dim=input_dim).to(device)
    else: 
        if task_labels_present == False: 
            model = FewShotRegressor_NoTaskLabel(input_dim=input_dim).to(device)
        else: 
            model = FewShotRegressor(input_dim=input_dim, task_label_dim=task_label_dim).to(device)
    
    # Initialize optimizer and loss function (same meta_lr used in MAML)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    if task_labels_present == True: 
        # Stage 1: Train Feature Extractor and Weights Generator with true task labels
        print("Starting Stage 1 training...")
        model = train(model, optimizer, criterion, device, num_epochs, use_task_labels=True, task_labels_present=True, sinusoid=sinusoid)
        
        # Stage 2: Train the entire model including Task Label Generator
        print("\nStarting Stage 2 training...")
        model = train(model, optimizer, criterion, device, num_epochs, use_task_labels=False, task_labels_present=True, sinusoid=sinusoid)
        
        if sinusoid == True: 
            torch.save(model.state_dict(), "model_TaskLabel_Sinusoid.pth")  # Save only state_dict
        else: 
            torch.save(model.state_dict(), "model_TaskLabel_Gaussian.pth")  # Save only state_dict
    
    else: 
        print("\nStarting Training...")
        model = train(model, optimizer, criterion, device, num_epochs, use_task_labels=False, task_labels_present=False, sinusoid=sinusoid) 

        if basic_maml == False: 
            if sinusoid==True: 
                torch.save(model.state_dict(), "model_NoTaskLabel_Sinusoid.pth")  # Save only state_dict
            else:  
                torch.save(model.state_dict(), "model_NoTaskLabel_Gaussian.pth")  # Save only state_dict
        else: 
            if sinusoid==True: 
                torch.save(model.state_dict(), "model_BasicMAML_Sinusoid.pth") 
            else:  
                torch.save(model.state_dict(), "model_BasicMAML_Gaussian.pth")  # Save only state_dict

    print("\n Training Completed! Model Saved!")
    
    # Test the model
    print("\nTesting model...")
    
    k_shot_test(model, optimizer, criterion, sinusoid, basic_maml=basic_maml, task_labels_present=task_labels_present, device=device)
    
    return model

# Sinusoid Dataset 
# Basic MAML (No Task Labels) 
#model_BasicMAML = train_full_model(task_labels_present=False, basic_maml=True, num_epochs=70000, sinusoid=True)

# Extensions 
#model_TaskLabel = train_full_model(task_labels_present=True, basic_maml=False, num_epochs=70000, sinusoid=True)
#model_NoTaskLabel  = train_full_model(task_labels_present=False, basic_maml=False,  num_epochs=70000, sinusoid=True)


# Gaussian 2D Dataset 
# Basic MAML (No Task Labels) 
model_BasicMAML = train_full_model(task_labels_present=False, basic_maml=True, num_epochs=1)

# Extensions 
model_TaskLabel = train_full_model(task_labels_present=True, basic_maml=False, num_epochs=1)
model_NoTaskLabel  = train_full_model(task_labels_present=False, basic_maml=False, num_epochs=1)

