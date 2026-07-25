from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        temp = list(text)
        hashSet = set(temp)
        vocab = list(hashSet)
        vocab.sort()
        stoi, itos = {char: i for i, char in enumerate(vocab)}, {i: char for i, char in enumerate(vocab)} # In Python 3.7+, the ordering from a list is maintained in a dict
        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        output = []
        for char in text:
            output.append(stoi[char])
        return output
    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        output = []
        for num in ids:
            output.append(itos[num])
        return "".join(output)

    """
    Notes:
    BPE is the tokenizer used by production LLMs, but character-level encoding, where every unique character gets its own integer is the simpler versoin. 
    A character-level vocab is the simplest tokenization approach, with vocab size equal to the numbero f unique characters in the training data. 
    The stoi/itos pair enables lossless round-trip conversion between text and integer sequences, which is a hard requirement for any tokenizer. 
    Sorting the unique characters ensures deterministic ID assignment. Without sorting, the same text could produce different vocabularies across runs. 
    """