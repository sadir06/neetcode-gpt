import numpy as np
from numpy.typing import NDArray
import math


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mew = np.mean(x) # Mean across features
        phi_squared = np.var(x) # We find the variance across the features
        epsilon = 1e-5

        x_hat = (((x - mew)/math.sqrt(phi_squared + epsilon) * gamma) + beta)
        return np.round(x_hat, decimals=5)
    
    """
    Notes:
    Layer normalizatoin fixes our values in the network going crazy high or small. 
    We re-centre and re-scale each layer's output so the values stay in a stable range. Every transformer block uses it twice, so you need it before building attention. 
    We learn parameters that could undo the normalisation because the network might need activations with a non-zero mean or non-unit variance for certain layers. The beta and gamma parameters give it that flexibility. 
    Layer norm is different to batch norm; layer norm normalizes across the feature dimension (each sample's statistics come from its own features), whereas batch norm normalises across the batch dimension (each feature's statistics come from the batch)
    The epsilon term might be small and seem insignificant, but prevents division by 0 when all features have the same value, ensuring that our model doesn't crash with a divide by 0 error. 
    """