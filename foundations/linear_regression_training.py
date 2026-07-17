import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        for _ in range(num_iterations):    
            prediction = self.get_model_prediction(X, initial_weights)
            N = len(X) 
            for j in range(len(initial_weights)):
                gradient = self.get_derivative(prediction, Y, N, X, j)
                initial_weights[j] -= self.learning_rate * gradient
        return np.round(initial_weights, 5)

    
    """
    Notes:
    This is the foundation of our Linear Regression Training. The pattern here (compute gradient per parameter, update each one) is exactly what optimiser.ste() does under the hood in PyTorch. The difference is that PyTorch automates the gradient computation with autograd. 
    Each weight is updated independently using hte gradient descent rule. This is called batch gradietn descent because we use all N samples for each update.
    Training computes the partial derivative of the loss with respect to EACH weight, which is just a dot product between the error vector and the feature column. 
    """
