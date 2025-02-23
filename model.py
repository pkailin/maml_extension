import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=2, feature_dim=40):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 40)
        self.fc2 = nn.Linear(40, feature_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Output is basis function representation
    
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim=64, ff_dim=128):
        super(AttentionBlock, self).__init__()
        """
        Attention block with self-attention, feed-forward layers, and layer normalization
        
        Args:
            embed_dim: Dimension of the embeddings
            ff_dim: Dimension of the first feed-forward layer
        """
        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=4)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x) # self attention so all Q, K, V are represented by the same embedding
        ff_output = self.ff_network(attn_output) # apply feed-forward network with residual connection and layer norm
        x = x + ff_output # add residual connection from input to output of second FC layer
        x = self.norm(x) # layer normalisation 
        return x

# Weights Generator with Self-Attention (With Task Labels)
class WeightsGenerator(nn.Module):
    def __init__(self, feature_dim=40, hidden_dim=64, task_label_dim=3):
        super(WeightsGenerator, self).__init__()
        """
        Weights Generator with multiple self-attention blocks
        
        Args:
            feature_dim: Dimension of the input features
            hidden_dim: Dimension of the hidden layers
            task_label_dim: Dimension of the task label
        """
        # Input is feature_dim + 1 (for labels) + task_label_dim
        self.total_input_dim = feature_dim + 1 + task_label_dim
        
        # Initial fully connected layer to get embeddings
        self.fc1 = nn.Linear(self.total_input_dim, hidden_dim)
        
        # Three attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, 128) for _ in range(3)
        ])
        
        # Final fully connected layer to get weights
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, features, labels, task_labels):
        """
        Forward pass through the Weights Generator
        
        Args:
            features: Input features of shape [batch_size, feature_dim]
            labels: Input labels (output values) of shape [batch_size, 1]
            task_labels: Task labels of shape [batch_size, task_label_dim]
            
        Returns:
            Weight vector of shape [feature_dim]
        """
        # Concatenate features, labels, and task labels
        x = torch.cat((features, labels, task_labels), dim=-1)
        
        # Pass through initial fully connected layer
        x = self.fc1(x)
        
        # Prepare for attention (convert to [seq_len, batch_size, hidden_dim])
        x = x.unsqueeze(1).transpose(0, 1)  # [1, batch_size, hidden_dim]
        
        # Pass through attention blocks
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # Convert back to [batch_size, hidden_dim]
        x = x.transpose(0, 1).squeeze(1)
        
        # Pass through final fully connected layer
        x = self.fc_out(x)
        
        # Take mean across batch to get final weight vector
        return x.mean(dim=0)
    
class WeightsGenerator_NoTaskLabel(nn.Module):
    def __init__(self, feature_dim=40, hidden_dim=64):
        super(WeightsGenerator_NoTaskLabel, self).__init__()

        # Input is feature_dim + 1 (for labels) 
        self.total_input_dim = feature_dim + 1 
        
        # Initial fully connected layer to get embeddings
        self.fc1 = nn.Linear(self.total_input_dim, hidden_dim)
        
        # Three attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, 128) for _ in range(3)
        ])
        
        # Final fully connected layer to get weights
        self.fc_out = nn.Linear(hidden_dim, feature_dim)
    
    def forward(self, features, labels):
        # Concatenate features, labels
        x = torch.cat((features, labels), dim=-1)
        
        # Pass through initial fully connected layer
        x = self.fc1(x)
        
        # Prepare for attention (convert to [seq_len, batch_size, hidden_dim])
        x = x.unsqueeze(1).transpose(0, 1)  # [1, batch_size, hidden_dim]
        
        # Pass through attention blocks
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # Convert back to [batch_size, hidden_dim]
        x = x.transpose(0, 1).squeeze(1)
        
        # Pass through final fully connected layer
        x = self.fc_out(x)
        
        # Take mean across batch to get final weight vector
        return x.mean(dim=0)


# Task Label Generator
class TaskLabelGenerator(nn.Module):
    def __init__(self, feature_dim=40, hidden_dim=64, task_label_dim=3):
        super(TaskLabelGenerator, self).__init__()
        self.total_input_dim = feature_dim + 1 
        
        # Initial fully connected layer to get embeddings
        self.fc1 = nn.Linear(self.total_input_dim, hidden_dim)
        
        # Three attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(hidden_dim, 128) for _ in range(3)
        ])
        
        # Final fully connected layer to get weights
        self.fc_out = nn.Linear(hidden_dim, task_label_dim)
    
    def forward(self, features, labels):
        
         # Concatenate features and labels 
        x = torch.cat((features, labels), dim=-1)
        
        # Pass through initial fully connected layer
        x = self.fc1(x)
        
        # Prepare for attention (convert to [seq_len, batch_size, hidden_dim])
        x = x.unsqueeze(1).transpose(0, 1)  # [1, batch_size, hidden_dim]
        
        # Pass through attention blocks
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        
        # Convert back to [batch_size, hidden_dim]
        x = x.transpose(0, 1).squeeze(1)
        
        # Pass through final fully connected layer
        x = self.fc_out(x)
        
        # Take mean across batch to get final task label (meaning batch must have same correct class label)
        x_mean = x.mean(dim=0)
        
        # Apply sigmoid only to the standard deviation (last dimension)
        # *** comment this line for sinusoid ***
        # x_mean[2] = torch.sigmoid(x_mean[2])  # Ensure std is positive
        
        return x_mean 

# Few-Shot Regression Model
class FewShotRegressor(nn.Module):
    def __init__(self, input_dim=2, feature_dim=40, task_label_dim=3):
        super(FewShotRegressor, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, feature_dim)
        self.weights_generator = WeightsGenerator(feature_dim, task_label_dim=task_label_dim)
        self.task_label_generator = TaskLabelGenerator(feature_dim, task_label_dim=task_label_dim)
    
    def forward(self, x_train, y_train, use_true_task_labels=False, true_task_labels=None):
        # Extract features
        features_train = self.feature_extractor(x_train)
        
        # Generate task labels using the Task Label Generator
        if use_true_task_labels and true_task_labels is not None:
            task_labels = true_task_labels
        else:
            task_labels = self.task_label_generator(features_train, y_train)
            task_labels = task_labels.repeat(10, 1) # assign the same task label to all 10 points in data 
        
        # Generate weights using features, labels and task labels
        weights = self.weights_generator(features_train, y_train, task_labels)
        
        # Make predictions using linear combination of basis functions
        return torch.matmul(features_train, weights)
    

# Few-Shot Regression Model (No Task Label)
class FewShotRegressor_NoTaskLabel(nn.Module):
    def __init__(self, input_dim=2, feature_dim=40):
        super(FewShotRegressor_NoTaskLabel, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, feature_dim)
        self.weights_generator = WeightsGenerator_NoTaskLabel(feature_dim)
    
    def forward(self, x_train, y_train):
        # Extract features
        features_train = self.feature_extractor(x_train)
        
        # Generate weights using features, labels 
        weights = self.weights_generator(features_train, y_train)
        
        # Make predictions using linear combination of basis functions
        return torch.matmul(features_train, weights)

# Basic MAML regressor 
class Regressor(nn.Module):
    def __init__(self, input_dim):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 40)  # First hidden layer
        self.fc2 = nn.Linear(40, 40)         # Second hidden layer
        self.fc_out = nn.Linear(40, 1)       # Output layer

    def forward(self, x_train):
        x = F.relu(self.fc1(x_train))  # First hidden layer with ReLU
        x = F.relu(self.fc2(x))  # Second hidden layer with ReLU
        x = self.fc_out(x)       # Output layer (regression task)
        return x