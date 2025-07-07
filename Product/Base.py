from abc import ABC, abstractmethod
import torch as tt

class Product(ABC):
    def __init__(self):
        pass
    
    def discounted_payoff(self, paths : tt.Tensor, r: float, T: float) -> tt.Tensor:
        pass