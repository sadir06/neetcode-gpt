import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.f1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        torch.manual_seed(0)
        x = self.f1(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = self.sigmoid(x)

        return torch.round(x, decimals=4)
        

    """
    Notes:
    Here we will build MNIST! One of the most basic neural nets, is a digit classifier, evey single ML Engineer has solved this. This is where deep learning started beating traditional algorithms. 
    We flatten the 28x28 piles into a flat tensor of 784 values. We have an input, hidden and output layer with 10 possible output. We can use droput to prevent overfitting (drops 50% of the digits during training).
    We have an inpuy 784 values, then a linear layer that projects 512, ReLU to add non-linearity, dropout to prevent overfitting (disable neurons from learning), and the final 10 sigmoid outputs (0-9) represent thye confidence that the image shows that digit. 
    """