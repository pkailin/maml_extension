from model import *
from data_generator import * 
from kshot_testing import * 
from trainer import * 

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def test_sinusoid(model, task_labels_present, basic_maml, path, device="cpu"):
    # Initialize optimizer and loss function
    criterion = nn.MSELoss()
    inner_lr = 0.001

    # Create test dataset with one task but more points for visualization
    test_dataset = SinusoidDataset(num_tasks=1, points_per_task=10)  
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Get adaptation data (first batch)
    x_train, y_train, task_labels = next(iter(test_loader))
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    # Get task parameters (amplitude and phase)
    unique_values = torch.unique(task_labels, dim=0).squeeze().tolist()
    amplitude, phase = unique_values
    
    model.train()
    # Quick adaptation (inner loop)
    for epoch in range(10):

        if basic_maml == True: 
            y_pred = model(x_train)
        elif task_labels_present:
            y_pred = model(x_train, y_train, use_true_task_labels=False, true_task_labels=None)
        else:
            y_pred = model(x_train, y_train)

        loss = criterion(y_pred.view(-1), y_train.view(-1))
        
        # Compute gradients manually
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

        # Perform manual gradient descent update
        with torch.no_grad():
            for param, grad in zip(model.parameters(), grads):
                param -= inner_lr * grad  # Manual update
    
    model.eval()

    # Generate smooth x values for plotting
    x_plot = torch.linspace(-5, 5, 200).reshape(-1, 1).to(device)

    # Get model predictions
    with torch.no_grad():

        if basic_maml == True: 
            y_pred = model(x_plot)
        
        else: 
            features_train = model.feature_extractor(x_train)
            features_plot = model.feature_extractor(x_plot)
            
            if task_labels_present == True: 
                # getting task label 
                task_labels = model.task_label_generator(features_train, y_train)
                task_labels = task_labels.repeat(10, 1) # assign the same task label to all points 

                # getting weights using training data 
                weights = model.weights_generator(features_train, y_train, task_labels)
            
            else: 
                weights = model.weights_generator(features_train, y_train) 
            
            # Make predictions using linear combination of basis functions
            y_pred = torch.matmul(features_plot, weights)
    
    # Calculate true sinusoid values
    true_y = amplitude * np.sin(x_plot.cpu().numpy() - phase)

    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot adaptation points
    plt.scatter(x_train.cpu().numpy(), y_train.cpu().numpy(), 
                color='red', label='Adaptation Points', s=50)
    
    # Plot true sinusoid
    plt.plot(x_plot.cpu().numpy(), true_y, 
             'b-', label=f'True Sinusoid (A={amplitude:.2f}, φ={phase:.2f})', 
             linewidth=2)
    
    # Plot predicted sinusoid
    plt.plot(x_plot.cpu().numpy(), y_pred.cpu().numpy(), 
             'g--', label='Model Prediction', 
             linewidth=2)
    
    plt.title('Sinusoid Function Adaptation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig(path[:-4]+ ".png")
    plt.close()

# For testing: 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Regressor(input_dim=1).to(device)  # Note: input_dim=1 for sinusoid
path = "model_BasicMAML_Sinusoid.pth"
model.load_state_dict(torch.load(path))
test_sinusoid(model, task_labels_present=False, basic_maml=True, path=path)
k_shot_test(model, sinusoid=True, basic_maml=True, device="cpu", task_labels_present=False)

'''
model = FewShotRegressor(input_dim=1, task_label_dim=2)
path = "model_TaskLabel_Sinusoid.pth"
model.load_state_dict(torch.load(path))
test_sinusoid(model, task_labels_present=True, basic_maml=False, path=path)
k_shot_test(model, sinusoid=True, basic_maml=False, device="cpu", task_labels_present=True)

model = FewShotRegressor_NoTaskLabel(input_dim=1)
path = "model_NoTaskLabel_Sinusoid.pth"
model.load_state_dict(torch.load(path))
test_sinusoid(model, task_labels_present=False, basic_maml=False, path=path)
k_shot_test(model, sinusoid=True, basic_maml=False, device="cpu", task_labels_present=False)
'''
