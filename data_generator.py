import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class GaussianDataset(torch.utils.data.Dataset):
    def __init__(self, num_tasks=1000, points_per_task=20, x_range=(-3, 3), y_range=(-3, 3)):
        self.num_tasks = num_tasks
        self.points_per_task = points_per_task
        
        # Arrays to store all data
        all_coords = []
        all_pdfs = []
        all_task_ids = []
        self.task_params = np.zeros((num_tasks, 3))  # [mean_x, mean_y, std] for each task
        
        for task_id in range(num_tasks):
            # Sample random mean and std for this task
            mean_x = np.random.uniform(-2, 2)
            mean_y = np.random.uniform(-2, 2)
            std = np.random.uniform(0.1, 2)
            self.task_params[task_id] = [mean_x, mean_y, std]
            
            # Sample random points for this task
            x = np.random.uniform(mean_x + x_range[0], mean_x + x_range[1], (points_per_task, 1))
            y = np.random.uniform(mean_y + y_range[0], mean_y + y_range[1], (points_per_task, 1))
            
            # Calculate 2D Gaussian probability density
            coords = np.hstack((x, y))
            dist_sq = ((x - mean_x)**2 + (y - mean_y)**2) / (2 * std**2)
            pdf = np.exp(-dist_sq) / (2 * np.pi * std**2)
            
            # Store data for this task
            all_coords.append(coords)
            all_pdfs.append(pdf)
            all_task_ids.extend([task_id] * points_per_task)
        
        # Combine all data
        self.coords = np.vstack(all_coords)
        self.pdfs = np.vstack(all_pdfs)
        self.task_ids = np.array(all_task_ids)
        
        # Convert to tensors
        self.coords = torch.tensor(self.coords, dtype=torch.float32)
        self.pdfs = torch.tensor(self.pdfs, dtype=torch.float32)
        self.task_ids = torch.tensor(self.task_ids, dtype=torch.long)
        self.task_params = torch.tensor(self.task_params, dtype=torch.float32)
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        # Return coordinates, pdf value, and task parameters
        task_id = self.task_ids[idx]
        return self.coords[idx], self.pdfs[idx], self.task_params[task_id]
    
class SinusoidDataset(torch.utils.data.Dataset):
    def __init__(self, num_tasks=1000, points_per_task=20, x_range=(-5, 5)):
        self.num_tasks = num_tasks
        self.points_per_task = points_per_task
        
        # Arrays to store all data
        all_x = []
        all_y = []
        all_task_ids = []
        self.task_params = np.zeros((num_tasks, 2))  # [amplitude, phase] for each task
        
        for task_id in range(num_tasks):
            # Sample random amplitude and phase for this task
            amplitude = np.random.uniform(0.1, 5.0)
            phase = np.random.uniform(0, np.pi)
            self.task_params[task_id] = [amplitude, phase]
            
            # Sample random x points for this task
            x = np.random.uniform(x_range[0], x_range[1], (points_per_task, 1))
            
            # Calculate sinusoid values: y = A * sin(x - phi)
            y = amplitude * np.sin(x - phase)
            
            # Store data for this task
            all_x.append(x)
            all_y.append(y)
            all_task_ids.extend([task_id] * points_per_task)
        
        # Combine all data
        self.x = np.vstack(all_x)
        self.y = np.vstack(all_y)
        self.task_ids = np.array(all_task_ids)
        
        # Convert to tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.task_ids = torch.tensor(self.task_ids, dtype=torch.long)
        self.task_params = torch.tensor(self.task_params, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        # Return input x, output y, and task parameters
        task_id = self.task_ids[idx]
        return self.x[idx], self.y[idx], self.task_params[task_id]