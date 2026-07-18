import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0) 
        std = math.sqrt(2.0/(fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        dims = [input_dim] + [hidden_dim] * num_layers
        weights = []
        for i in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2.0 / (dims[i] + dims[i + 1]))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / dims[i])
            else:
                std = 1.0 # This is if we don't use anything, we just use a standard normal distribution
            w = torch.randn(dims[i + 1], dims[i]) * std
            weights.append(w)
        x = torch.randn(1, input_dim)
        stds = []
        for w in weights:
            x = x @ w.T # Transpose
            x = torch.relu(x) # same as np.maximum
            stds.append(round(x.std().item(), 2))

        return stds

    """
    Notes:
    If we try to stack 10 layers with a normal distribution weights -> the activations will either explore to inifnite or collapse to 0. So, we just have to initialise our weights properply!
    This is why we can use Xavier or Kaiming initialisatoin! These set the std so that teach layer preserves the variance of its input. ReLU zeros out roughly half the neurons so that the variance drops by half at every layer, and Kaiming compensates by doubling the numerator relative to Xavier. 
    
    
    """