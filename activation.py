import numpy as np

class ActivatioNFunction:
    def __init__(self):
        pass

class ReLu(ActivatioNFunction):
    def __init__(self):
        super().__init__()
        self.name = "ReLu"
    
    def __call__(self, x):
        return np.maximum(0, x)
    
class Sigmoid(ActivatioNFunction):
    def __init__(self):
        super().__init__()
        self.name = "Sigmoid"
    
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class Tanh(ActivatioNFunction):
    def __init__(self):
        super().__init__()
        self.name = "Tanh"
    
    def __call__(self, x):
        return np.tanh(x)

class Softmax(ActivatioNFunction):
    def __init__(self):
        super().__init__()
        self.name = "Softmax"
    
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

class Linear(ActivatioNFunction):
    def __init__(self):
        super().__init__()
        self.name = "Linear"
    
    def __call__(self, x):
        return x
