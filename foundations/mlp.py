import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        # Given an input, weights and biases, output a 1D numpy array, the network output. 
        inp = x
        for i, weight in enumerate(weights):
            l1 = inp @ weight + biases[i]
            l2 = np.maximum(l1, 0)
            inp = l2 # Set the input for the next layer to the output of the first before looping again
        return np.round(inp, decimals=5)


    """
    Notes:
    MLP isjust connecting layers of neurons end to end. The output of one layer feeds into the input of the next. 
    Mathematically, it's just repeated matrix multiplication with activations in between, but stacking layers lets the network learn patterns that a single neuron never could. 
    ReLU is actually only applied on hidden layers, we don't want it affecting our output layer.
    """