from model import *
from data_generator import * 
from kshot_testing import * 
from trainer import * 

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
'''

# Function to test and plot Gaussian distribution estimation
def test_gaussian_pdf(model, task_labels_present, path, basic_maml, device="cpu"):
    # Initialize optimizer and loss function (same meta_lr used in MAML)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    test_dataset = GaussianDataset(num_tasks=1, points_per_task=200) # first 10 for adaptation, the rest for testing 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    x_train, y_train, task_labels = next(iter(test_loader)) 
    unique_values = torch.unique(task_labels, dim=0).squeeze().tolist()
    # Assign to variables
    mean_x, mean_y, std = unique_values

    model.train()
    for epoch in range(10): # 10 num_updates for quick adaptation 
        optimizer.zero_grad()
        # use generated task labels in second stage 
        if task_labels_present:
            y_pred = model(x_train, y_train, use_true_task_labels=False, true_task_labels=None)
        else:
            y_pred = model(x_train, y_train)  # basic maml

        loss = criterion(y_pred.view(-1), y_train.view(-1))
        loss.backward()
        optimizer.step()
    
    model.eval()  # Set model to evaluation mode
    # Initialize lists to collect the remaining batches
    x_test = []
    y_test = []

    # Iterate through the remaining batches
    for x, y, _ in test_loader:
        x_test.append(x)
        y_test.append(y)

    # Concatenate the remaining batches into a single tensor
    x_test = torch.cat(x_test, dim=0)
    y_test = torch.cat(y_test, dim=0)

    # Extract the first column (all rows, column index 0)
    x_range = x_test[:, 0].tolist()
    # Extract the second column (all rows, column index 1)
    y_range = x_test[:, 1].tolist()

    X, Y = np.meshgrid(x_range, y_range)
    
    test_coords = np.vstack([X.ravel(), Y.ravel()]).T  # Flatten grid to [N, 2]
    test_coords_tensor = torch.tensor(test_coords, dtype=torch.float32).to(device)
    empty_tensor = torch.empty(test_coords_tensor.shape[1], 1)
    
    # Predict the estimated PDF using the model
    with torch.no_grad():

        if basic_maml == True: 
            y_pred = model(test_coords_tensor, empty_tensor).cpu().numpy()
        
        else: 
            features_train = model.feature_extractor(x_train)
            features_plot = model.feature_extractor(test_coords_tensor)
            
            if task_labels_present == True: 
                # getting task label 
                task_labels = model.task_label_generator(features_train, y_train)
                task_labels = task_labels.repeat(test_coords_tensor.shape[1], 1) # assign the same task label to all points 

                # getting weights using training data 
                weights = model.weights_generator(features_plot, empty_tensor, task_labels)
            
            else: 

                weights = model.weights_generator(features_plot, empty_tensor) 
            
            # Make predictions using linear combination of basis functions
            y_pred = torch.matmul(features_plot, weights).cpu().numpy()
    
    estimated_pdf = y_pred.reshape(X.shape)  # Reshape to match grid

    # Compute the true Gaussian PDF for comparison
    dist_sq = ((X - mean_x)**2 + (Y - mean_y)**2) / (2 * std**2)
    true_pdf = np.exp(-dist_sq) / (2 * np.pi * std**2)

    # Plot the results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # True Gaussian PDF
    ax[0].contourf(X, Y, true_pdf, levels=20, cmap="viridis")
    ax[0].set_title("True Gaussian PDF")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    # Estimated PDF from the model
    ax[1].contourf(X, Y, estimated_pdf, levels=20, cmap="viridis")
    ax[1].set_title("Estimated PDF")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.savefig(path[:-4]+ ".png")
'''


# Function to test and plot Gaussian distribution estimation
def test_gaussian_pdf(model, task_labels_present, path, basic_maml, device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    test_dataset = GaussianDataset(num_tasks=1, points_per_task=200)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    x_train, y_train, task_labels = next(iter(test_loader))
    unique_values = torch.unique(task_labels, dim=0).squeeze().tolist()
    mean_x, mean_y, std = unique_values  # Extract Gaussian parameters

    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        y_pred = model(x_train, y_train, use_true_task_labels=False, true_task_labels=None) if task_labels_present else model(x_train, y_train)

        loss = criterion(y_pred.view(-1), y_train.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()

    # Create a 2D grid covering the Gaussian range
    x_min, x_max = -3+mean_x, 3+mean_x
    y_min, y_max = -3+mean_y, 3+mean_y
    grid_resolution = 100  # Higher = smoother plot

    x_range = np.linspace(x_min, x_max, grid_resolution)
    y_range = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_range, y_range)

    test_coords = np.vstack([X.ravel(), Y.ravel()]).T  # Convert grid to [N, 2]
    test_coords_tensor = torch.tensor(test_coords, dtype=torch.float32).to(device)

    # Ensure `empty_tensor` has the same batch size as `test_coords_tensor`
    empty_tensor = torch.empty(test_coords_tensor.shape[0], 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        if basic_maml:
            y_pred = model(test_coords_tensor, empty_tensor).cpu().numpy()
        else:
            features_train = model.feature_extractor(x_train)
            features_plot = model.feature_extractor(test_coords_tensor)

            if task_labels_present:
                task_labels = model.task_label_generator(features_train, y_train)
                task_labels = task_labels.expand(test_coords_tensor.shape[0], -1)  # Expand to match batch size
                weights = model.weights_generator(features_plot, empty_tensor, task_labels)
            else:
                weights = model.weights_generator(features_plot, empty_tensor)

            y_pred = torch.matmul(features_plot, weights).cpu().numpy()

    estimated_pdf = y_pred.reshape(X.shape)  # Reshape to match grid

    # Compute the true Gaussian PDF
    dist_sq = ((X - mean_x)**2 + (Y - mean_y)**2) / (2 * std**2)
    true_pdf = np.exp(-dist_sq) / (2 * np.pi * std**2)

    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].contourf(X, Y, true_pdf, levels=20, cmap="viridis")
    ax[0].set_title("True Gaussian PDF")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    ax[1].contourf(X, Y, estimated_pdf, levels=20, cmap="viridis")
    ax[1].set_title("Estimated PDF")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    plt.savefig(path[:-4]+ ".png")

# For testing: 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Regressor(input_dim=2).to(device)  # Note: input_dim=1 for sinusoid
path = "model_BasicMAML_Gaussian.pth"
model.load_state_dict(torch.load(path))
test_gaussian_pdf(model, task_labels_present=False, basic_maml=True, path=path)

model = FewShotRegressor_NoTaskLabel(input_dim=2)
path = "model_NoTaskLabel_Gaussian.pth"
model.load_state_dict(torch.load(path))
test_gaussian_pdf(model, task_labels_present=False, basic_maml=False, path=path)

model = FewShotRegressor(input_dim=2, task_label_dim=3)
path = "model_TaskLabel_Gaussian.pth"
model.load_state_dict(torch.load(path))
test_gaussian_pdf(model, task_labels_present=True, basic_maml=False, path=path)


