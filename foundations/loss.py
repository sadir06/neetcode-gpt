import numpy as np
from numpy.typing import NDArray


class Solution:
    # Cross Entropy Loss is all about telling the model how wrong it was. 
    # It compares the predicted distribution to the actual answer and produces a single number. The lower the loss, the better.
    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        # This is specifically when we have 2 classes
        # We have 2 inputs, true labels vs predicted labels
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        L = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return np.round(L, decimals=4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        L = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return np.round(L, decimals=4)
