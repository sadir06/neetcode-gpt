import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        return torch.reshape(to_reshape, (-1, 2)) # We can use -1 here in a tuple as a trick, it tells pytorch to just generate as many rows as needed, that doesn't matter, but make sure to generate 2 columns.

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.mean(to_avg, dim=0) # Takes the column-wise mean, average across all rows

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.cat((cat_one, cat_two), dim=1) # This kind of glues the 2 tensors in any dimension that you want. 

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.nn.functional.mse_loss(prediction, target) # This just returns the MSE loss, very basic. 



    """
    Notes:
    Pretty much any serious AI model uses PyTorch because it makes interaction with tensors, and especially GPUs really easy. This is great for taking use of incredibly powerful GPUs to train and run models on and incredibly fast speeds. 
    There's a lot of underlying math that PyTorch just abstracts away, such as backpropagation (autograd), but it's a good idea to understand the maths behind it anyway. 
    These 4 operations are incredibly important to learn know:
    Reshape -> Converts an M x N trnsor into a shape (M @ N // 2) x 2 (flattening the elements then folding into 2 columns)
    Average -> Reduce a 2D tensor along dimension 0, returning the per-column mean.
    Concatenate -> Join an M x N tensor with an M x M tensor side-by-side -> M x (M + N).
    MSE Loss: Computes the mean squared error between a prediction vector and a target vector.
    """