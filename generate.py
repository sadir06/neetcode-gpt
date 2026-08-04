import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = []
        for i in range(new_chars):
            if context.shape[1] > context_length:
                context = context[:, -context_length:] # crop context
            x = model(context)
            last_logits = x[:, -1, :]
            probs = nn.functional.softmax(last_logits, dim=-1)

            next_token = torch.multinomial(probs, 1, generator=generator)
            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            # MORE OF YOUR CODE (arbitrary number of lines)
            context = torch.cat((context, next_token), dim=-1)
            result.append(int_to_char[next_token.item()])
        
        return "".join(result)

        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!




"""
Notes:
We now train the model to speak. We have an autoregressive loop where the model generates one token at a time, appends it to the context, and repeats. This is the inference-time procedure that turns a trained language model into a text generator. The generation lok works like this: crop the context to the model's maximum context length (if it is too long), feed the context through the model to get probabilities at every position (using softmax), and extract tthe last positoin, as only the final position's distributions matters since it predicts the next token. We then draw a token from the distribution using torch.multinomial. Then we add the sampled token to the context, and we convert the token ID into a character. Then we repeat this for the number of desided new cahracters. 

While we could take the argmax (most probably token every time), that produces deterministic and repetitve text. Samping from the full distrivution introduces a variety. In production, temperature scaling (probs = softmax(logits/T)) and top-k filtering control the creativity-coherence tradeoff. Also, the once context exceeds the max length C, the earlier tokens are croppsed off. The model's memory is limited to the most recent C tokens. This is why GPT models have a context length limit. 
"""