class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x), where the learning rate is usually 0.01
        # Round final answer to 5 decimal places
        # We hve a number of iterations that determines how many gradient descent steps to execute, which could be 0. 
        # We also have init, which can be any number and is the starting point for the optimisation
        x = init
        for _ in range(iterations):
            x = x - learning_rate * 2 * (x) # This is applying the update rule onto the initial given number 

        return round(x, 5) # Rounds to 5 decimal places
        
        """
        Notes:
        Gradient Desent finds the minima of the function, and is scaled up or down by the learning rate. 
        The learning rate is one of the most important hyperparameters. Too large = divergence. Too small = slow convergence. 
        While for convex functions, the gradient descent converges to the globabl minimum, in practice, the neural net loss surfaces are not convex, but gradient descent still works well in practice. 
        """
