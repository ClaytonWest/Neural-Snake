import numpy as np

class Genome():
  def __init__(self):
    self.fitness = 0
# Sets neuron count in hidden layer
    hidden_layer = 10
# initalizing the weights
# Generates a random value from normal distribution 
    self.w1 = np.random.randn(6, hidden_layer)
    self.w2 = np.random.randn(hidden_layer, 20)
    self.w3 = np.random.randn(20, hidden_layer)
    self.w4 = np.random.randn(hidden_layer, 3)

# Implements forward pass of neural network
# Computes weighted sum of inputs with the weights being (w.1,w.2, w.3, w.4)
# relu function sets negative values to zero and keep positive values
  def forward(self, inputs):
    net = np.matmul(inputs, self.w1)
    net = self.relu(net)
    net = np.matmul(net, self.w2)
    net = self.relu(net)
    net = np.matmul(net, self.w3)
    net = self.relu(net)
    net = np.matmul(net, self.w4)
    net = self.softmax(net)
    return net

# Rectifier function, positive argument
  def relu(self, x):
    return x * (x >= 0)
# Predicts possible outcomes
  def softmax(self, x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
  
# Activation function (.01 = training coefficient)
  def leaky_relu(self, x):
    return np.where(x > 0, x, x * 0.01)
