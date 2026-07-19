import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        # Given x, gamma and epsilon, we need to output a list of floats, which is hte normalised output
        
        x = np.array(x)
        gamma = np.array(gamma)

        mean = np.mean(np.square(x))
        x_hat = x / (np.sqrt(mean + eps))

        return np.round(x_hat * gamma, decimals=4).tolist()

    
    """
    Notes:
    Modern LLMs don't use layer/bach norm. Instead of these, we use RMS Normalization, which drops the mean subtraciton entirely. Instead of subtracting the mean, dividing by standorddeviation and scaling ans shifting, we skip the subtraction and remove beta entirely. We only need the root mean square. 
    This means fewer parameters and less memory. Since we are also doing less operations, we are a lot faster too. 



    """