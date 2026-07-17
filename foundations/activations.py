import numpy as np
from numpy.typing import NDArray


class Solution:
    # Activation functions give neural nets the ability to learn complext patterns.
    # Without them it's just a series of linear transformations, the output will just be a linera function of the input, no matter how many layesr you stack. 
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        # Given a 1D numpy array, we have to use the formula to return the probability using the sigmoid function
        return np.round(1/(1 + np.exp(-z)), decimals=5) # np.exp(x) = e^x

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        # Given an input numpy array z, we have to return 0 for negative inputs and the input itself for positive values
        return np.maximum(0, z)
    """
    Notes:
    We have these tricks because we don't want the model to just learn linear behaviours, we want it to understand more complex correlations
    Activation functions introduce non-linearity by introducing a non - linear bench after each linear step
    The Sigmoid squashes any real number into the range (0, 1). It's like a "confidence meter", where large positive values map close to 1, and large negative values map close to 0 and 0 maps to exampctly 0.5 -? This makes it natural for binary classification with output layers where you want a probabiltiy.
    However, this introduces the vanishing gradient problem: when abs(z) is large, the derivative of it is nearly 0, so gradients shrink into nothing during backprop and deep layers stop learning. We have GeLU for ReLU for this exact reason.
    The ReLU is the default actvation function because it's easy to compute, nad produces sparse activations. Its weakness is "dying ReLU": if a neuron always receives negative input, it's gradient is always 0 and nevery updates.  
    """