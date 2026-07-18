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

        error = y_pred - y_true # dL/dy (error signal)
        sigmoid_deriv = (1 - y_pred) * (y_pred) # dy_pred/dz (Signoid derivative)
        # dz/dw = x (the input)
        delta = error * sigmoid_deriv # The delta is the product of the first 2, and tells us "how wrong we are, scaled by how sensitive the ctivation is at this operating point."

        dL_dw = np.round(delta * x, 5) # We use these derivaties to calculate the backpropagation!
        dL_db = round(float(delta), 5)
        return (dL_dw, dL_db)


    """
    Notes:
    Backpropagation is the same idea as gradient descent applied to a neuron: figure out how much each weight contributed to the error, then nudege it in hte direction that reduces the loss. 
    It's just the chain rule, applied one link at a time form the output back to each weight. 
    The delta term is the core building block. In deeper networks, deltas propagate backward through layers. 
    The sigmoud derivative peaks at 0.25 when Y = 0.5 and approaches 0 at the extremes, which is why deep sigmoud networks suffer from vanihsing gradients.
    The main function to remember here is dL/dw = dL/dy_pred * dy_pred/dz * dz/dw 
    """