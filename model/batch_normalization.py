import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        # They didn't give us numpy arrays we have to change everything
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)
        epsilon = 1e-5

        if not momentum or momentum == 0:
            momentum = 0.1
        if training:  
            mean = np.mean(x, axis=0) # Mean, make sure that it is on axis 0 so that wedo batch norm, axis=1 is layer norm. 
            var = np.var(x, axis=0) # Var
            x_hat = ((x - mean) / np.sqrt(var + epsilon)) # Normalise
            # We also need the running statistics for use during training
            running_mean = np.multiply((1 - momentum), running_mean) + momentum * mean
            running_var = np.multiply((1 - momentum), running_var) + momentum * var
        else: # Inference
            x_hat = ((x - running_mean) / np.sqrt(running_var + epsilon))
        y = gamma * x_hat + beta # Scale and shift!

        return (np.round(y, 4).tolist(), np.round(running_mean, 4).tolist(), np.round(running_var, 4).tolist())


    """
    Notes:
    Batch Normalization normalizes across the batch for each features. This makes training faster and more stable, especially in convolutional and fully-connected networks. 
    Think about batch norm vs layer norm, as normalising each column vs normalising each row. In batch norm, we don't care about any other features except the current one, while in layer norm, we consider one row from every single feature. 
    We compute the mean, the variance and then normalise it. We use gamma and beta to scale and shift it as necessary.
    """