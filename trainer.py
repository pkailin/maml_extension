import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from data_generator import *
import copy 

def train(model, criterion, device, num_epochs, use_task_labels, basic_maml=False, task_labels_present = True, inner_lr=1e-3, outer_lr=1e-3, num_inner_updates=1, sinusoid=False):

    # Initialize meta-optimizer (Adam) for outer loop
    meta_optimiser = optim.Adam(model.parameters(), lr=outer_lr)

    # initialise models
    model_inner = copy.deepcopy(model)
    model_outer = copy.deepcopy(model)
    best_valid_model = copy.deepcopy(model)
    best_valid_loss = np.inf

    for epoch in range(num_epochs):

        # copy params of network
        copy_params = [w.clone() for w in model_outer.parameters()]

        # initialise cumulative gradient
        meta_gradient = [0 for _ in range(len(copy_params))]


        num_tasks = 25
        if sinusoid == False: 
            train_dataset = GaussianDataset(num_tasks=num_tasks, points_per_task=20) # 10 for inner gradient update, 10 for outer gradient update, 10 for validation  
        else: 
            train_dataset = SinusoidDataset(num_tasks=num_tasks, points_per_task=20)
        
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

            # reset network weights
            for param_inner, param_copy in zip(model_inner.parameters(), copy_params):
                param_inner.data.copy_(param_copy.data)
        
            # Inner loop updates
            for _ in range(num_inner_updates):
                
                if task_labels_present == True: 
                    if use_task_labels == True: 
                        y_pred = model_inner(x_train, y_train, use_true_task_labels=True, true_task_labels=task_labels_train)
                    else: 
                        y_pred = model_inner(x_train, y_train, use_true_task_labels=False, true_task_labels=None)
                elif basic_maml == True:
                    y_pred = model_inner(x_train)
                else: 
                    y_pred = model_inner(x_train, y_train)

                loss_task = criterion(y_pred.view(-1), y_train.view(-1))

                # compute the gradient wrt current model
                params = [w for w in model_inner.parameters()]
                grads = torch.autograd.grad(loss_task, params, create_graph=True, retain_graph=True, allow_unused=True)

                # Perform manual weight update using inner_lr
                # make an update on the inner model using the current model (to build up computation graph)
                for param, grad in zip(model_inner.parameters(), grads):
                    if grad is not None:
                        param.data = param.data - inner_lr * grad
    
            # ------------ compute meta-gradient on test loss of current task ------------

            if basic_maml == True:
                y_pred = model_inner(x_test)
            else: 
                features_train = model_inner.feature_extractor(x_train)
                features_test = model_inner.feature_extractor(x_test)
                
                if task_labels_present == True: 
                    if use_task_labels == True: 
                        #task_labels_train = task_labels_test
                        task_labels = task_labels_test
                    else: 
                        task_labels = model_inner.task_label_generator(features_train, y_train)
                        task_labels = task_labels.repeat(10, 1) 
                    # get weights 
                    weights = model_inner.weights_generator(features_train, y_train, task_labels)
                else: 
                    weights = model_inner.weights_generator(features_train, y_train)

                y_pred = torch.matmul(features_test, weights)
            
            loss_meta = criterion(y_pred.view(-1), y_test.view(-1))
            
            # compute gradient w.r.t. outer model
            task_grads = torch.autograd.grad(loss_meta, model_outer.parameters(), allow_unused=True)
            for i in range(len(meta_gradient)):
                if task_grads[i] is not None:  # Safety check for None gradients
                    meta_gradient[i] += task_grads[i].detach()
        
        # ------------ meta update ------------
        meta_optimiser.zero_grad()

        # assign meta-gradient
        for param, meta_grad in zip(model_outer.parameters(), meta_gradient):
            if meta_grad is not None:
                param.grad = torch.tensor(meta_grad / num_tasks, device=device).expand_as(param)

        # Clear meta gradients for next iteration
        meta_gradient = [torch.zeros_like(p) for p in model_outer.parameters()]

        # do update step on outer model
        meta_optimiser.step()

        if (epoch%10 == 0): 
            # ------------ logging ------------
            # evaluate on validation set (any task)

            if sinusoid == False: 
                valid_dataset = GaussianDataset(num_tasks=1, points_per_task=20) # 10 to get weights, 10 to calculate loss 
            else: 
                valid_dataset = SinusoidDataset(num_tasks=1, points_per_task=20)
            
            # Sample tasks
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, 
                batch_size=20,  # Total points needed per task
                shuffle=False
            )

            for x_batch, y_batch, task_labels in valid_loader:

                # Split into train and test
                x_train, y_train, task_labels_train = x_batch[:10], y_batch[:10], task_labels[:10]
                x_test, y_test, task_labels_test = x_batch[10:], y_batch[10:], task_labels[10:]

                if basic_maml == True:
                    y_pred = model_outer(x_test)
                else: 
                    features_train = model_outer.feature_extractor(x_train)
                    features_test = model_outer.feature_extractor(x_test)
                    
                    if task_labels_present == True: 
                        if use_task_labels == True: 
                            #task_labels_train = task_labels_test
                            task_labels = task_labels_test
                        else: 
                            task_labels = model_outer.task_label_generator(features_train, y_train)
                            task_labels = task_labels.repeat(10, 1) 
                        # get weights 
                        weights = model_outer.weights_generator(features_train, y_train, task_labels)
                    else: 
                        weights = model_outer.weights_generator(features_train, y_train)

                    y_pred = torch.matmul(features_test, weights)

            loss_valid = criterion(y_pred.view(-1), y_test.view(-1))

            if loss_valid < best_valid_loss: 
                best_valid_model = copy.copy(model_outer)
                best_valid_loss = loss_valid

            print('Current Epoch: ' + str(epoch)+ ', Current Best Validation Loss: ' + str(best_valid_loss.item()))
    
    return best_valid_model