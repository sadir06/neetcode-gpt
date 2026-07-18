import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        
        y = np.dot(x, w) + b # Elementwise dot product and addition

        if activation == "sigmoid":
            return np.round(1/(1 + np.exp(-y)), decimals=5)
        else: # relu
            return np.round(np.maximum(0, y), 5)

        """
        Notes:
        A neuron/perceptron is the simplest unit in a neural network. it mimics a biological neuron, receving signals, processing them, and firing. 
        The Sigmoid activation squishes any real number into the range(0, 1)
        The ReLU is an activation function that simply clips any values below 0 back up to 0. This add's non-linearity into the model.
        Stacking many neurons into layers is how we build deep neural networks. 
        The dot product measures similarity between the input and the weight vector. Training adjusts w so that this similarity is high for inputs the neuron should activate on. 
        """

