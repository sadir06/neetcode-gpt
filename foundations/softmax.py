import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # Given a vector of raw scores (logits), we have to convert them into a probability distribution
        # Output values are all positive and sum to 1, making them interpretable as probabilities
        # We can use np.max, np.exp and np.sum here
        softmax = (np.exp(z - np.max(z))) / np.sum(np.exp(z - max(z)))

        return np.round(softmax, decimals=4)