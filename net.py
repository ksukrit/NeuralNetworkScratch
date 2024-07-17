import numpy as np 
from activation import *
from layers import *
from loss import *
from data import *

class Net:
    def __init__(self, loss):
        self.Layer1 = Linear(784, 256)
        self.Layer2 = Linear(256, 128)
        self.Layer3 = Linear(128, 64)
        self.Layer4 = Linear(64, 10)
        self.activation1 = ReLu()
        self.activation2 = ReLu()
        self.activation3 = ReLu()
        self.activation4 = Softmax()
        self.loss = loss

    def forward(self, x):
        x = self.Layer1(x)
        x = self.activation1(x)
        x = self.Layer2(x)
        x = self.activation2(x)
        x = self.Layer3(x)
        x = self.activation3(x)
        x = self.Layer4(x)
        x = self.activation4(x)
        return x

    def backward(self, grad):
        grad = self.activation4.backward(grad)
        grad = self.Layer4.backward(grad)
        grad = self.activation3.backward(grad)
        grad = self.Layer3.backward(grad)
        grad = self.activation2.backward(grad)
        grad = self.Layer2.backward(grad)
        grad = self.activation1.backward(grad)
        grad = self.Layer1.backward(grad)
        return grad

    def update(self, lr):
        self.Layer1.update(lr)
        self.Layer2.update(lr)
        self.Layer3.update(lr)
        self.Layer4.update(lr)

    def train(self, x, y, lr):
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        grad = self.loss.backward(y_pred, y)
        self.backward(grad)
        self.update(lr)
        return loss

    def test(self, x, y):
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        return loss

    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)


if __name__ == "__main__":
    dataset = Dataset("mnist/")
    train_loader = DataLoader(dataset, 64)
    test_loader = DataLoader(dataset, 64, shuffle=False)
    net = Net(CrossEntropy())
    for epoch in range(10):
        for x, y in train_loader:
            loss = net.train(x, y, 0.01)
        print(f"Epoch: {epoch}, Loss: {loss}")
    correct = 0
    for x, y in test_loader:
        y_pred = net.predict(x)
        correct += np.sum(y_pred == y)
    print(f"Accuracy: {correct / len(dataset)}")