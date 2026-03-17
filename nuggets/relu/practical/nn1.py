import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets


class ReLU:
    def forward(self, x):
        self.old_x = np.copy(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        return np.where(self.old_x > 0, grad, 0)


class Sigmoid:
    def forward(self, x):
        self.old_y = 1.0 / (1.0 + np.exp(-x))
        return self.old_y

    def backward(self, grad):
        return self.old_y * (1.0 - self.old_y) * grad


class Softmax:
    def forward(self, x):
        shifted = x - x.max(axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.old_y = exp_x / exp_x.sum(axis=1, keepdims=True)
        return self.old_y

    def backward(self, grad):
        return self.old_y * (grad - (grad * self.old_y).sum(axis=1, keepdims=True))


class CrossEntropy:
    def forward(self, x, y):
        self.old_x = x.clip(min=1e-8, max=None)
        self.old_y = y
        return np.where(y == 1, -np.log(self.old_x), 0).sum(axis=1)

    def backward(self):
        return np.where(self.old_y == 1, -1.0 / self.old_x, 0)


class Linear:
    def __init__(self, n_in, n_out):
        self.weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        self.biases = np.zeros(n_out)

    def forward(self, x):
        self.old_x = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad):
        self.grad_b = grad.mean(axis=0)
        self.grad_w = np.dot(self.old_x.T, grad) / self.old_x.shape[0]
        return np.dot(grad, self.weights.T)


class Model:
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def loss(self, x, y):
        return self.cost.forward(self.forward(x), y)

    def backward(self):
        grad = self.cost.backward()
        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)


net = Model(
    [Linear(784, 100), ReLU(), Linear(100, 10), Softmax()],
    CrossEntropy()
)


def train(model, lr, nb_epoch, data):
    for epoch in range(nb_epoch):
        running_loss = 0.0
        num_inputs = 0

        for inputs, targets in data:
            num_inputs += inputs.shape[0]

            running_loss += model.loss(inputs, targets).sum()
            model.backward()

            for layer in model.layers:
                if isinstance(layer, Linear):
                    layer.weights -= lr * layer.grad_w
                    layer.biases -= lr * layer.grad_b

        print(f"Epoch {epoch+1}/{nb_epoch}: loss = {running_loss / num_inputs:.4f}")


def load_minibatches(batch_size=64):
    tsfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    trn_set = datasets.MNIST('.', train=True, download=True, transform=tsfms)
    trn_loader = torch.utils.data.DataLoader(
        trn_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    data = []
    for inputs_t, targets_t in trn_loader:
        inputs = inputs_t.view(inputs_t.size(0), -1).numpy()
        targets = np.zeros((targets_t.size(0), 10))
        targets[np.arange(targets_t.size(0)), targets_t.numpy()] = 1.0
        data.append((inputs, targets))

    return data

if __name__ == "__main__":
    data = load_minibatches(batch_size=64)
    data = data[:10]
    train(net, lr=0.1, nb_epoch=1, data=data)