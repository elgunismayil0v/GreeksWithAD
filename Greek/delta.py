from Pricer.Pricer import MonteCarloPricer
import torch as tt

def compute_delta(pricer : MonteCarloPricer, S0: tt.Tensor, NoOfPaths: int, NoOfSteps: int) -> tt.Tensor:
    price = pricer.price(S0, NoOfPaths, NoOfSteps)
    delta = tt.autograd.grad(price, S0, create_graph=True)[0]
    return delta



