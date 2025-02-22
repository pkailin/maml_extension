import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_generator import *

class replace_params:
    """
    Context manager to temporarily replace model parameters
    """
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.original_params = [p.clone() for p in model.parameters()]
        
    def __enter__(self):
        for param, new_param in zip(self.model.parameters(), self.params):
            param.data.copy_(new_param.data)
    
    def __exit__(self, *args):
        for param, original_param in zip(self.model.parameters(), self.original_params):
            param.data.copy_(original_param.data)

def train(model, optimizer, criterion, device, num_epochs, use_task_labels, task_labels_present = True, inner_lr=1e-3, num_inner_updates=1, sinusoid=False):


    for epoch in range(num_epochs):

        # Copy weights of network (outer model)
        model.train()
        copy_params = [p.clone() for p in model.parameters()]
        
        # Initialize meta gradient accumulator
        meta_gradient = [torch.zeros_like(p) for p in model.parameters()]

        if sinusoid == False: 
            train_dataset = GaussianDataset(num_tasks=25, points_per_task=20) # 10 for inner gradient update, 10 for outer gradient update 
        else: 
            train_dataset = SinusoidDataset(num_tasks=25, points_per_task=20)
        
        # Sample tasks
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=20,  # Total points needed per task
            shuffle=False
        )
        
        # For each task in meta-update
        for batch_idx, (x_batch, y_batch, task_labels) in enumerate(train_loader):

            # Split into train and test
            x_train, y_train, task_labels_train = x_batch[:10], y_batch[:10], task_labels[:10]
            x_test, y_test, task_labels_test = x_batch[10:], y_batch[10:], task_labels[10:]
            
            # Initialize inner model with copied parameters
            inner_params = [p.clone().requires_grad_(True) for p in copy_params]
            
            # Inner loop updates
            for _ in range(num_inner_updates):
                
                # Forward pass with current inner parameters
                with replace_params(model, inner_params):
                    if task_labels_present == True: 
                        if use_task_labels == True: 
                            y_pred = model(x_train, y_train, use_true_task_labels=True, true_task_labels=task_labels_train)
                        else: 
                            y_pred = model(x_train, y_train, use_true_task_labels=False, true_task_labels=None)
                    else: 
                        y_pred = model(x_train, y_train)

                    loss_task = criterion(y_pred.view(-1), y_train.view(-1))
                
                # Compute gradients for inner update
                grads = torch.autograd.grad(
                    loss_task, 
                    inner_params,
                    create_graph=False,
                    retain_graph=True, 
                    allow_unused=True
                )
                
                # Update inner parameters (handling None gradients)
                new_inner_params = []
                for param, grad in zip(inner_params, grads):
                    if grad is not None:
                        new_inner_params.append(param - inner_lr * grad)
                    else:
                        new_inner_params.append(param.clone())
                inner_params = new_inner_params
            
            # Compute meta-gradient using test set
            with replace_params(model, inner_params):
                if task_labels_present == True: 
                    if use_task_labels == True: 
                        y_pred = model(x_test, y_test, use_true_task_labels=True, true_task_labels=task_labels_test)
                    else: 
                        y_pred = model(x_test, y_test, use_true_task_labels=False, true_task_labels=None)
                else: 
                    y_pred = model(x_test, y_test)

                loss_meta = criterion(y_pred.view(-1), y_test.view(-1))
            
            # Compute gradient w.r.t. original parameters
            task_grads = torch.autograd.grad(loss_meta, copy_params, allow_unused=True)
            
            # Accumulate meta-gradients
            for i in range(len(meta_gradient)):
                if task_grads[i] is not None:
                    meta_gradient[i] += task_grads[i].detach()
        
        # Meta update
        optimizer.zero_grad()
        
        # Assign accumulated meta-gradients
        for param, meta_grad in zip(model.parameters(), meta_gradient):
            param.grad = meta_grad / 25
        # Perform meta-optimization step
        optimizer.step()

        if (epoch%1000 == 0): 
            print(f"Epoch {epoch+1}, Meta Loss: {loss_meta.item():.6f}")
    
    return model