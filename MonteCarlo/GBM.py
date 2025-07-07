from Base import MonteCarloSimulator
import torch as tt

class GBM(MonteCarloSimulator):
    def __init__(self, T : float, r : float, sigma : float, device : str = 'cpu'):
        """Geometric Brownian Motion (GBM) simulator for financial modeling.
        Parameters:T : float The time horizon for the simulation.
        r : float The risk-free interest rate.
        sigma : float The volatility of the underlying asset."""
        self.T = T
        self.r = r
        self.sigma = sigma
        self.device = device
    
    def simulate(self, S0 : tt.tensor, NoOfPaths : int, NoOfSteps : int) -> tt.Tensor:
        """
        Simulate the paths of a Geometric Brownian Motion (GBM).

        Parameters:
        S0 : float The initial asset price.
        NoOfPaths : int The number of paths to simulate.
        NoOfSteps : int The number of time steps in each path.

        Returns:
        torch.Tensor A tensor of shape (M, N+1) containing the simulated paths.
        """
        dt = self.T / NoOfSteps
        S = tt.zeros((NoOfPaths, NoOfSteps + 1), dtype=tt.float64)
        S[:, 0] = S0
        for i in range(1, NoOfSteps + 1):
            Z = tt.randn(NoOfPaths, dtype=tt.float64, device=self.device)
            S[:, i] = S[:, i - 1] * tt.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * tt.sqrt(dt) * Z)
            
        return S