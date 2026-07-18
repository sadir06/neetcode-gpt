import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x, w) + b 
        y_pred = 1/(1 + np.exp(-z))

        error = y_pred - y_true
        sigmoid_deriv = (1 - y_pred) * (y_pred)
        delta = error * sigmoid_deriv

        dL_dw = np.round(delta * x, 5)
        dL_db = round(float(delta), 5)
        return (dL_dw, dL_db)


    """
    Notes:
    Backpropagation is the same idea as gradient descent applied to a neuron: figure out how much each weight contributed to the error, then nudege it in hte direction that reduces the loss. 
    It's just the chain rule, applied one link at a time form the output back to each weight. 


    """