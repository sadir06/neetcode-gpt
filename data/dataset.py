import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        dataset = raw_dataset.split()
        ix = torch.randint(low=0, high = len(dataset) - context_length, size = (batch_size, )).tolist()
        X = []
        Y = []
        for i in ix:
            X.append(dataset[i : i + context_length])
            Y.append(dataset[i + 1: i + context_length + 1])
        print(X)
        print(Y)

        return X, Y

"""
Notes:
Now, how do we go from raw text to finalised training data? We need to first get a bunhc of data from the internet, and then a text corupus gets sliced into overlapping (context, next_token) pairs using a sliding window. For a document of length L with context length C, any position [0, L - C - 1] (0-indexed) yeilds:
X = [tokeni, ..., tokeni + C - 1], Y = [tokeni + 1, ..., tokeni + C]


"""
