from abc import ABC
import torch as tt
class MonteCarloSimulator(ABC):
    def __init__(self):
        pass
    
        def simulate(self, S0 : tt.tensor, NoOfPaths : int, NoOfSteps : int) -> tt.Tensor:
            pass