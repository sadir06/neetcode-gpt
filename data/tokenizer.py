from typing import List
from collections import Counter

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        merges = []
        for _ in range(num_merges):
            if len(tokens) < 2:
                break # Only one pair
            pairs = Counter(zip(tokens[:-1], tokens[1:]))

            if not pairs:
                berak

            best_count = max(pairs.values())
            candidates = sorted(p for p, c in pairs.items() if c == best_count)
            best = candidates[0] # lexiographically the largest pair

            merges.append([best[0], best[1]]) # this can be multiple letters in future iterations of num_merges

            # Merge all non-overlapping occurrences left to right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best[0] and tokens[i + 1] == best[1]:
                    new_tokens.append(best[0] + best[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return merges

    """
    Notes:
    Byte Pair Ecoding solves the problem of the model not encountering words it has never seen before. It learns a volcabulary of SUBWORD tokens. GPT, LLaMa and most modern LLMs use BPE tokenization!
    Split the corups into individual characters, count the frequency of every adjacent pair of tokens,nad merge hte most frequeient pair inot a new token. Replace all non-overlapping occurrences of htat pair, and repeat. 
    Each merge creates a new token, and common words like "the" become single tokens after just a few merges. Rare words stay as subword pices, which is exactly what we want, where common patterns get compressed and rare words decompose into known subcomponents. 
    GPT-2 uses about 50,000 merges, with a vocabulary of 50,257 tokens. The encoding process replays the merges in order to tokenize any input text. 
    This is actually a greedy algorithm, because we merge only the most frequent pairs, achieving a vocabulary htat compresses common patterns while decomposing rare words into known pieces. 
    Non-overlapping left-to-right merging ensures deterministic, reproducible results. 
    BPE elmiminates out-of-vocab problems: any input can be encoded as a sequence of subword tokens, even words the model has never seen before. 
    """
