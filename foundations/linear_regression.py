import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        result = X @ weights # Dot Product betweeen the feature maxtrix and the model weights
        return np.round(result, decimals=5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        error = (ground_truth - model_prediction)
        MSE = np.mean(error * error)
        return np.round(MSE, decimals=5)
