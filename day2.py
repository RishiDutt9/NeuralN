import numpy as np
import matplotlib.pyplot as plt

#Define activation functions

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

#Forward pass Function

def forward_pass(X,weights,biases,activation_functions):
    z = np.dot(weights,X) + biases
    a = activation_functions(z)
    return a

#Example input
X = np.array([[0.5],[0.8]])
weights = np.array([[0.2,0.4],[0.6,0.1]])
biases = np.array([[0.1],[0.2]])

#perform forward pass with Diffrent Activation Functions

activations = {
    "Sigmoid":sigmoid,
    "Tanh":tanh,
    "ReLU":relu,
    "Softmax":softmax
}

for name ,func in activations.items():
    output = forward_pass(X,weights,biases,func)
    print(f"Activation Function : {name} \nOutput : {output}\n\n")


#Define Range of inputs
z = np.linspace(-10,10,100)
#Ploting Activation Functions

plt.figure(figsize=(10,5))
plt.plot(z,sigmoid(z),label="Sigmoid")
plt.plot(z,tanh(z),label="Tanh")
plt.plot(z,relu(z),label="ReLU")
plt.plot(z,softmax(z),label="Softmax")
plt.title("Activation Functions")
plt.xlabel("Input is (z)")
plt.ylabel("Output of Activation Function")
plt.grid(True)
plt.legend()
plt.show()