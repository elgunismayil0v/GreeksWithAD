from MonteCarlo.GBM import GBM
from Product.EuropeanCall import EuropeanCall
from Pricer.Pricer import MonteCarloPricer
from Greek.delta import compute_delta
from Greek.gamma import compute_gamma
import torch

# Use double precision (important for financial computations)
torch.set_default_dtype(torch.float64)

# --- Parameters ---
S0_val = 100.0            # Initial stock price
r = 0.05                  # Risk-free rate
sigma = 0.2               # Volatility
T = 1.0                   # Maturity in years
K = 100                   # Strike price
NoOfSteps = 100           # Time steps in simulation
NoOfPaths = 100           # Number of Monte Carlo paths
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Components ---
model = GBM(T, r, sigma, device=device)
product = EuropeanCall(K)
pricer = MonteCarloPricer(model, product, r, T)

# --- Pricing ---
S0_val = 100.0  # example spot price
S0_price = torch.tensor(S0_val, dtype=torch.float64, requires_grad=True)
price = pricer.price(S0_price, NoOfPaths, NoOfSteps)
print(f"Option Price: {price.item():.4f}")

# --- Greeks ---

# Use fresh S0 tensors for gradient computations
S0_delta = torch.tensor(S0_val, dtype=torch.float64, requires_grad=True)
delta = compute_delta(pricer, S0_delta, NoOfPaths, NoOfSteps)
print(f"Delta: {delta.item():.4f}")

#S0_gamma = torch.tensor(S0_val, dtype=torch.float64, requires_grad=True)
#gamma = compute_gamma(pricer, S0_gamma, NoOfPaths, NoOfSteps)
#print(f"Gamma: {gamma:.4f}")


