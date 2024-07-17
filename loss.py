import numpy as np

class Loss:
    def __init__(self) -> None:
        pass
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred, y_true):
        raise NotImplementedError
    
    def backward(self):
        raise NotImplementedError
    
class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))
    
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]

class CrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred))
    
    def backward(self, y_pred, y_true):
        return -y_true / y_pred / y_true.shape[0]

class BinaryCrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, y_pred, y_true):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred, y_true):
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]