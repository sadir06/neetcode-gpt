import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        # Given an input of 2 strings (one expressing positive emotions and the other as negative), place all positive sentence rows first, then negative sentence rows. 
        combined = positive + negative
        vocab = sorted({word for sentence in combined for word in sentence.split()})
        word_to_id = {word : idx + 1 for idx, word in enumerate(vocab)}

        encoded = [torch.tensor([word_to_id[w] for w in s.split()]) for s in combined]

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)

    """
    Notes:
    Word embeddings give each token a dense vector, but we need to first decide what counts as a token. Tokenization is all about splitting raw text into pices and assigning each piece an integer ID. This is the first stpe of the NLP pipeline, text goes in, and a sequence of integers comes out. 
    NLP preprocessing converts variable-length strings into fixed-size numerical tensors that neural networks can process. 
    Vocab IDs starta at 1 so that 0 serves as a padding token, allowing models to mask and ignore padding positions. 
    Sorting the vocab ensures deterministic ID assignment, which is critical for reproductibility and testing. 

    """