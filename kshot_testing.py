import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import os

import torch
import copy 
from data_generator import *

def k_shot_test(model, optimizer, criterion, sinusoid, basic_maml, device="cpu", task_labels_present=True):
    # Store initial model state
    original_state = copy.deepcopy(model.state_dict())
    losses_per_step = [[] for _ in range(11)]  # 0-10 steps (including initial)
    
    for t in range(600):  # Loop over 600 different tasks
        # Reset model to initial state
        model.load_state_dict(original_state)
        
        # Generate datapoints for one task
        if sinusoid == False:
            test_dataset = GaussianDataset(num_tasks=1, points_per_task=20)
        else:
            test_dataset = SinusoidDataset(num_tasks=1, points_per_task=20)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)
        x_data, y_data, _ = next(iter(test_loader))
        
        # Split into support (training) and query (test) sets
        x_train, y_train = x_data[:10].to(device), y_data[:10].to(device)
        x_test, y_test = x_data[10:].to(device), y_data[10:].to(device)
        
        # Compute initial loss (before adaptation)
        with torch.no_grad():
            if basic_maml == True:
                y_pred = model(x_test, y_test)
            else:
                features_train = model.feature_extractor(x_train)
                
                if task_labels_present == True: 
                    # getting task label 
                    task_labels = model.task_label_generator(features_train, y_train)
                    task_labels = task_labels.repeat(10, 1) # assign the same task label to all 10 points in data 

                    # getting weights using training data 
                    weights = model.weights_generator(features_train, y_train, task_labels)
                
                else: 
                    weights = model.weights_generator(features_train, y_train) 

                features_test = model.feature_extractor(x_test)
                # Make predictions using linear combination of basis functions
                y_pred = torch.matmul(features_test, weights)
            
            initial_loss = criterion(y_pred.view(-1), y_test.view(-1))
            losses_per_step[0].append(initial_loss.item())
        
        # Adaptation steps
        for step in range(10):
            optimizer.zero_grad()
            
            # Forward pass on training set
            if task_labels_present:
                y_pred = model(x_train, y_train, use_true_task_labels=False, true_task_labels=None)
            else:
                y_pred = model(x_train, y_train)
            
            # Compute loss and adapt
            loss = criterion(y_pred.view(-1), y_train.view(-1))
            loss.backward()
            optimizer.step()
            
            # Evaluate on test set after adaptation
            with torch.no_grad():
                if basic_maml == True:
                    y_pred = model(x_test, y_test)
                else:
                    features_train = model.feature_extractor(x_train)
                    
                    if task_labels_present == True: 
                        # getting task label 
                        task_labels = model.task_label_generator(features_train, y_train)
                        task_labels = task_labels.repeat(10, 1) # assign the same task label to all 10 points in data 

                        # getting weights using training data 
                        weights = model.weights_generator(features_train, y_train, task_labels)
                    
                    else: 
                        weights = model.weights_generator(features_train, y_train) 

                    features_test = model.feature_extractor(x_test)
                    # Make predictions using linear combination of basis functions
                    y_pred = torch.matmul(features_test, weights)
                    
                test_loss = criterion(y_pred.view(-1), y_test.view(-1))
                losses_per_step[step + 1].append(test_loss.item())
    
    # Compute statistics
    losses_mean = [np.mean(step_losses) for step_losses in losses_per_step]
    losses_conf = [
        st.t.interval(0.95, len(step_losses) - 1, loc=np.mean(step_losses), scale=st.sem(step_losses))
        if len(step_losses) > 1 else (np.nan, np.nan)
        for step_losses in losses_per_step
    ]
    
    # Print results
    print("\n=== K-Shot Test Results: Loss per Adaptation Step ===\n")
    for step in range(11):  # 0-10 steps
        mean_loss = np.round(losses_mean[step], 4)
        conf_interval = np.round(losses_conf[step], 4)
        print(f"Step {step}: Loss = {mean_loss:.4f} ± {np.abs(conf_interval[1] - mean_loss):.4f}")

