import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential # Sequential model provides layers one after the other
from tensorflow.keras.layers import  Dense, Dropout, Flatten, Conv2D, MaxPooling2D
#Dense : Fully connected layer
#Dropout : Regularization technique
#Flatten : Flatten the input,Flatten multi-dimensional input to 1D
#Conv2D : Convolutional layer, cov2d is used to create a convolutional layer applied over the input layer to fetch the features out of it.
#MaxPooling2D : Pooling layer, MaxPooling2D is used to create a pooling layer, which is used to reduce the spatial dimensions of the output volume.

# Load the MNIST data

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Normalize the data

X_train = X_train.reshape(-1,28,28,1).astype('float32')/255.0
X_test = X_test.reshape(-1,28,28,1).astype('float32')/255

#One hot encode the labels

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print(f"Train Data Shape : {X_train.shape}")
print(f"Test Data Shape : {X_test.shape}")

#Build the model
model = Sequential([
Conv2D(filters=32, 
       kernel_size=(3,3), 
       activation='relu', 
       input_shape=(28,28,1),
       padding='same'),
       MaxPooling2D(pool_size=(2,2)),
       Flatten(),
       Dense(128, activation='relu'),
       Dropout(0.5),
       Dense(10, activation='softmax') 

])
#Flatten() is used to flatten the input,Flatten multi-dimensional input to 1D
#MaxPooling2D it will reduce each 2 x 2 to a single maximum value. This is done to reduce the spatial dimensions of the output volume.
#Filters learnable features detectors
#Kernel_size is the size of the filter matrix for the convolution
#Activation function is used to introduce non-linearity in the model
#Input_shape is the shape of the input data


#Display the model Architecture

model.summary()

#Compile the model

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#Trrain the Model

model.fit(X_train, y_train, 
    batch_size=32,
    epochs=10,
    validation_split=0.2)


#Evaluate the model

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss : {test_loss:.4f},\n Test Accuracy : {test_acc:.4f}")

#Save the model

model.save('mnist_Classifier.h5')

#Load the model
from tensorflow.keras.models import load_model
loaded_model = load_model('mnist_Classifier.h5')

#Verify the model performance
loss, acc = loaded_model.evaluate(X_test, y_test)
print(f"Lodded model Loss : {loss:.4f},\n Loaded model Accuracy : {acc:.4f}")
