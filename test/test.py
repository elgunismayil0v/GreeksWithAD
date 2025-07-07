import torch
from MonteCarlo.GBM import GBM
from Product.EuropeanCall import EuropeanCall
from Pricer.Pricer import MonteCarloPricer

def test_call_price_positive():
    model = GBM(T=1.0, r=0.05, sigma=0.2, device='cpu')
    product = EuropeanCall(K=100)
    pricer = MonteCarloPricer(model, product, r=0.05, T=1.0)

    price = pricer.price(torch.tensor(100.0, requires_grad=True), NoOfPaths=50, NoOfSteps=100)
    assert price.item() > 0