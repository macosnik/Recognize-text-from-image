import numpy as np

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i + 1], layers[i]) * 0.1
            b = np.zeros((layers[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        self.activations = [x]
        self.z_list = []
        
        activation = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            activation = self.sigmoid(z)
            self.z_list.append(z)
            self.activations.append(activation)
        
        return activation
    
    def backward(self, x, y, lr):
        m = x.shape[1]
        
        delta = (self.activations[-1] - y) * self.sigmoid_derivative(self.activations[-1])
        
        for i in range(len(self.layers) - 2, -1, -1):
            dw = np.dot(delta, self.activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
            
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(self.activations[i])
    
    def train(self, x, y, epochs, lr):
        for epoch in range(epochs):
            output = self.forward(x)
            self.backward(x, y, lr)
            loss = np.mean((output - y) ** 2)
            print(f"\rEpoch {epoch}, Loss: {loss:.6f}", end="")
    
    def save_model(self, path):
        model_dict = {
            'layers': self.layers,
            'weights': self.weights,
            'biases': self.biases
        }
        np.savez(path, **model_dict)
    
    def load_model(self, path):
        data = np.load(path)
        self.layers = data['layers']
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])