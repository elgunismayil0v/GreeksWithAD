from Base import Product
import torch as tt

class EuropeanCall(Product):
    def __init__(self, K : float):
        """European Call Option product.
        
        Parameters:
        K : float The strike price of the option.
        """
        self.K = K
        
        
    def discounted_payoff(self, paths : tt.Tensor, r: float, T: float) -> tt.Tensor:
        """
        Calculate the discounted payoff of a European Call option.

        Parameters:
        paths : torch.Tensor A tensor of shape (M, N+1) containing the simulated paths.
        r : float The risk-free interest rate.
        T : float The time to maturity.

        Returns:
        torch.Tensor A tensor of shape (M,) containing the discounted payoffs.
        """
        NoOfPaths = paths.shape[0]
        discount_payoff = tt.clamp(paths[:, -1] - self.K, min=0) / NoOfPaths * tt.exp(-r * T)
        return discount_payoff
        