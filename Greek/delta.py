from MonteCarlo import MonteCarloPricer
import torch as tt

def compute_delta(pricer: MonteCarloPricer, S0_val: float, NoofPaths: int, NoofSteps: int) -> float:
    S0 = tt.tensor(S0_val, requires_grad=True, dtype=tt.float64)
    price = pricer.price(S0, NoofPaths, NoofSteps)
    price.backward()
    return S0.grad.item()

