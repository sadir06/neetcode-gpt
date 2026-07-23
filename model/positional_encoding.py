import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        # Given an input of the dimension of the encoding and the number of positions, we should return a 2D array of shape seq_len, d_model. 
        PE = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1, 1) # (seq_len, 1)
        div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
        PE[:, 0::2] = np.sin(position / div_term)
        PE[:, 1::2] = np.cos(position / div_term[:PE[:, 1::2].shape[1]])
        return np.round(PE, 5)

    """
    Notes:
    Transformers process all tokens in parallel. Unlike RNs, they don't read left-to-right. Without position information, the sentence "dog bites man" would be identical to "man bites dog". Positional encoding solves this by injecting positional information into the embeddings using sine and cosine waves at different frequencies. 
    Sinusoidal positional encoding injects position information using the sine and cosine patterns at geometrically spaced frequencies. 
    Different frequencies capture different position scales: high freqiencies distinguish nearby tokens, low frequencies distinguish distant ones. 
    Positional encodings are added to word embeddings, so they must have the same dimension as the embedding vectors. 
    """