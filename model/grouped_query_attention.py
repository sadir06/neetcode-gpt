import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        # Scaled dot-product attention with a causal mask
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask==0, float("-inf")) # Set future values to negative infinity so that they just dissappear and attention doesn't take them into account
        weights = torch.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return torch.round(self.output_proj(out), decimals=4)
"""
Notes:
While multi-head attentoin gives every head it's own KV projects, during inference with KV cache and storing KV for all heads is very expensive. Grouped Query Attention shares KV across groups of heads, achieving the same quality with a fraction of the memory. 

The way it works is that we are in a middle ground of MHA vs MQA, one head vs multiple heads. We land in a niddle ground where we have g heads where 1 < g < h, so that the cache size is g/h of MHA and the quality is still near MHA. Each KV head serves a group of h / g query heads. We project Q with h heads but K and V with only g heads, and then expand the g KV heads to match the h query heads by repeating each KV head within the group. After this expansion, attention proceeds exactly as standard MHA: scaled dot-product with causal masking. E.g. for 8 Q heads and 2 KV heads, each KV heads is shared by 4 query heads. The expansion repeats each of the 2 KV head vectors 4 times to produce 8 matching vectors. 
"""