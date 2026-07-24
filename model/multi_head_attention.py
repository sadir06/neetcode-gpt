import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        self.att_heads = nn.ModuleList()
        for i in range(num_heads):
            self.att_heads.append(self.SingleHeadAttention(embedding_dim, attention_dim // num_heads))
        self.output_proj = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Run each head on the input, concatenate outputs along dim=2
        # Pass concatenated result through the output projection (W_O)
        # Return result rounded to 4 decimal places
        head_output = []
        for head in self.att_heads:
            head_output.append(head(embedded))
        concatenated = torch.cat(head_output, dim=2)
        return torch.round(self.output_proj(concatenated), decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v

    """
    Notes:
    A signle attention head can only focus on one type of relationship, but obviously language has many patterns. So we use multi-headed attention, which runs several attentoin heads in paralle, with each learning different patterns, and then combines them. GPT-3 uses 96 heads working simultaenously. 
    Wo is a learned output projection. Each head independently learns which tokens to attend to. Their outputs are concatenated and passed through a final linear projection that mixes information across heads. 
    Multi-headed attention runs several attention heads in parallel, with a learned output projection that combined their outputs. 
    Each head operates on a d/h dimensional subspace, and concatenation reconstructs the full dimension, making it a drop-in replacement for single head attention. 
    Using nn.ModuleList (instead of a plain python list) is essential here os htat PyTorch can track and update each head's parameters during training. 
    Essentially, given h heads, nad dimension d, we create h heads each with dimension d/h, na deach run indepoendently on the same input. Then we concatenate all the head outputs along the feature dimension. Then we apply an output projection Wo to combine the heads. 
    This way different heads learn different htings. Some heaads learn syntactic relationships, some learn semantic relationships, and some learn positional patterns. A single head would have to compromise between all these patterns, which multiple heads specialise. 
    """
