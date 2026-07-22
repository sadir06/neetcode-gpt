import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        # Given training data x and targets y, perform gradient descent for a specified number of epochs.
        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        for _ in range(epochs): # We will iterate this many times
            y_hat = np.dot(X, w) + b

            error = (y_hat - y) 

            w = w - lr * ((2/n) * (X.T @ error))
            b = b - lr * ((2/n) * np.sum(error))

        return (np.round(w, 5), round(float(b), 5))




    """
    Notes:
    Training time! Everything is useless unless the model actually learns from the data. 
    The training loop pattern (forward, loss, backward, update) is universal across all gradient-based models, from linear regression to billion-parameter transformers.
    Vectorized gradient computation using X.T @ error replaces the per-weight loops, turning O(d @ N) separate dot produucts into a single matrix multiply.
    Initialising the weights to 0s works for linear regressoin but causes symmetry problems in deeper networks, where random initialisation is needed to break the symmetry!
    """