import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        # Given our embeddings table, and token_ids, we have to output a 2D array of shape (len(token_ids), embed_dim)
        embeddings.tolist()
        token_ids.tolist()
        output = [[] for _ in range(len(token_ids))]
        for i, token_id in enumerate(token_ids):
            output[i] = embeddings[token_id]  

        matrix = np.array(output)

        return np.round(matrix, 5)
    """
    Notes:
    Here, we'll start processing text with neural nets, from raw words down to the positional encodings used in transformers. 
    Word embeddings amap each token to a dense vector where similar words end up close together. These embeddings emerge through training, and similar words get grouped closer and closer together. This is incredibly useful to "teach" the model how to form correlations with different words. 

    """