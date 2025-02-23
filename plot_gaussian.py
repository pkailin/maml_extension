from model import *
from data_generator import * 
from kshot_testing import * 
from trainer import * 

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Function to test and plot Gaussian distribution estimation
def test_gaussian_pdf(model, task_labels_present, path, basic_maml, device="cpu"):
    criterion = torch.nn.MSELoss()
    inner_lr = 0.001

    test_dataset = GaussianDataset(num_tasks=1, points_per_task=200)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

    x_train, y_train, task_labels = next(iter(test_loader))
    unique_values = torch.unique(task_labels, dim=0).squeeze().tolist()
    mean_x, mean_y, std = unique_values  # Extract Gaussian parameters

    model.train()
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

    # Create a 2D grid covering the Gaussian range
    x_min, x_max = -3+mean_x, 3+mean_x
    y_min, y_max = -3+mean_y, 3+mean_y
    grid_resolution = 100  # Higher = smoother plot

    x_range = np.linspace(x_min, x_max, grid_resolution)
    y_range = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x_range, y_range)

    test_coords = np.vstack([X.ravel(), Y.ravel()]).T  # Convert grid to [N, 2]
    test_coords_tensor = torch.tensor(test_coords, dtype=torch.float32).to(device)


    with torch.no_grad():
        if basic_maml:
            y_pred = model(test_coords_tensor).cpu().numpy()

        else:
            features_train = model.feature_extractor(x_train)
            features_plot = model.feature_extractor(test_coords_tensor)

            if task_labels_present == True: 
                # getting task label 
                task_labels = model.task_label_generator(features_train, y_train)
                task_labels = task_labels.expand(10, 1)  # 10 to match training set size 
                weights = model.weights_generator(features_train, y_train, task_labels)
                
            else:
                weights = model.weights_generator(features_train, y_train) 

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


