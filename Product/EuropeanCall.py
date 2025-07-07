from Base import Product
import torch as tt

class EuropeanCall(Product):
    def __init__(self, K : float):
        """European Call Option product.
        
        Parameters:
        K : float The strike price of the option.
        """
        self.K = K
        
        
    def payoff(self, paths : tt.Tensor) -> tt.Tensor:
        """
        Calculate the  payoff of a European Call option.

        Parameters:
        paths : torch.Tensor A tensor of shape (M, N+1) containing the simulated paths.

        Returns:
        torch.Tensor A tensor of shape (M,) containing the payoffs.
        """
        payoff = tt.clamp(paths[:, -1] - self.K, min=0)
        return payoff
        