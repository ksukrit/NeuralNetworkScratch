import numpy as np

class Layer:
    def __init__(self) -> None:
        pass
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def update(self, lr):
        pass

class Dropout(Layer):
    def __init__(self, p) -> None:
        super().__init__()
        self.p = p
    
    def forward(self, x):
        self.mask = np.random.binomial(1, self.p, size=x.shape) / self.p
        return x * self.mask
    
    def backward(self, grad):
        return grad * self.mask
    
    def update(self, lr):
        pass

class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        if bias:
            self.bias = np.zeros(out_features)
    
    def forward(self, x):
        self.x = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad):
        self.grad_weight = np.dot(self.x.T, grad)
        self.grad_bias = np.sum(grad, axis=0)
        return np.dot(grad, self.weight.T)
    
    def update(self, lr):
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias
