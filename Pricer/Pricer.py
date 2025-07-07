from Product.Base import Product
from MonteCarlo.Base import MonteCarloSimulator
import torch as tt
import numpy as np

class MonteCarloPricer:
    def __init__(self, model : MonteCarloSimulator, product : Product, r : float, T: float):
        """Monte Carlo pricer for financial products.
        
        Parameters:
        model : MonteCarloPricer The Monte Carlo model used for pricing.
        product : Product The financial product to be priced.
        r : float The risk-free interest rate.
        T : float The time to maturity in years.
        """
        self.model = model
        self.product = product
        self.r = r
        self.T = T
        
    def price(self, S0 : tt.Tensor, NoOfPaths : int, NoOfSteps : int) -> tt.Tensor:
        """
        Price the financial product using Monte Carlo simulation.
        
        Parameters:
        NoOfPaths : int The number of simulated paths.
        NoOfSteps : int The number of time steps in each path.
        
        Returns:
        torch.Tensor A tensor containing the price of the product.
        """
        paths = self.model.simulate(S0, NoOfPaths, NoOfSteps)
        payoffs = self.product.payoff(paths)
        discount_factor = tt.exp(
    tt.tensor(-self.r * self.T, dtype=payoffs.dtype, device=payoffs.device)
)
        discounted_payoffs = discount_factor * payoffs

        price = tt.mean(discounted_payoffs)
        return price