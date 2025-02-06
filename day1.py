import tensorflow as tf
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Load CIFAR-10 dataset
cifar10 = tf.keras.datasets.cifar10
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

print("MNIST dataset loaded:")
print(f"Training data shape: {x_train_mnist.shape}, Training labels shape: {y_train_mnist.shape}")
print(f"Test data shape: {x_test_mnist.shape}, Test labels shape: {y_test_mnist.shape}")

print("CIFAR-10 dataset loaded:")
print(f"Training data shape: {x_train_cifar10.shape}, Training labels shape: {y_train_cifar10.shape}")
print(f"Test data shape: {x_test_cifar10.shape}, Test labels shape: {y_test_cifar10.shape}")


#Define a Dense layer
layer = tf.keras.layers.Dense(units = 10,activation = 'relu')
print(f"Tensorflow layer: {layer}")

#define a Basic layer in PyTorch
layer = nn.Linear(in_features = 10,out_features= 10)

#Visual Sample

plt.imshow(x_train_mnist[0], cmap='gray')
plt.title(f"MNIST Label: {y_train_mnist[0]}")
plt.show()

#Visual CIFAR-10 Sample
plt.imshow(x_train_cifar10[0])
plt.title(f"CIFAR-10 Label: {y_train_cifar10[0]}")
plt.show()


