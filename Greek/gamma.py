import torch as tt
from Greek.delta import compute_delta
import torch as tt
from Greek.delta import compute_delta

def compute_gamma(pricer, S0 : tt.Tensor, NoOfPaths: int, NoOfSteps: int) -> float:
    # Step 1: Compute option price
    price = pricer.price(S0, NoOfPaths, NoOfSteps)  # Must return scalar and depend on S0

    # Step 2: Compute first derivative (delta)
    delta = tt.autograd.grad(price, S0, create_graph=True)[0]

    # Step 3: Compute second derivative (gamma)
    gamma = tt.autograd.grad(delta, S0)[0]

    return gamma.item()

