import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if kv_cache is None:
            kv_cache = KVCache()
        
        full_k, full_v = kv_cache.update(k, v)

        scores = (q @ full_k.transpose(-2, -1)) * (full_k.shape[-1] ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ full_v

        return torch.round(output, decimals=4), kv_cache
"""
Notes:
Now, our GPT generates text, but it's incredibly slow, because each new token recomputes attention for ALL previous tokens. This is, as you can expect, incredibly inefficient. Instead of this, we can apply a KV-cache to fix this (as fetching from cache is infinitely faster than running mult-headed attention many times). 

Basically during autoregressive generation (one new token at a time), each new token runs a full forward pass. Inside each attention layer, the model computes Q, K, V from the ENTIRE context. This is O(N^2), as QKV are all computer for all N tokens, and the attention matrix is N x N. Generating 100 tokens means recomputing 1 + 2 + 3 + .... + 100 = 5050 times across all steps. However, since KV for previous tokens don't change between generation steps, and only the new token's K and V need computing, instead of recomputing all N key and value vectors, we cache the ones from previous steps and append only the new token's K and V (an absolutely TINY cost)! This turns the attention computation from O(N^2) to just O(N)! The new query only needs to attend over the growing cache, not recompute everything. 
"""