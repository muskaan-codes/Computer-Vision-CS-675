# MNIST Large ConvNet Demo for CS 672 at UMass Boston

from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Load MNIST datasets for training (60,000 exemplars) and testing (10,000 exemplars)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)

# Normalize inputs to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

num_epochs = 200
batch_size = 100

size_input = (28, 28, 1)                # This time we have two-dimensional input arrays representing 28x28 pixel intensity values
size_output = len(np.unique(y_train))   # Number of output-layer neurons must match number of classes

num_train_exemplars = x_train.shape[0] 

# Build the model (computational graph)
mnist_model = Sequential(
    [keras.Input(shape=size_input, name='input_layer'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_1'),
    Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_2'),
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_1'),
    Dropout(0.3),
    Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_3'),
    Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='ReLU', name='conv_4'),
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_2'),
    Dropout(0.3),
    Flatten(name='flat_layer'),
    Dense(200, activation='ReLU', name='dense_layer_1'),
    Dropout(0.3),
    BatchNormalization(),
    Dense(size_output, activation='softmax', name='output_layer')])

# Print a summary of the model's layers, including their number of neurons and weights (parameters) in each layer
mnist_model.summary()

mnist_model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics='accuracy')

mnist_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=2)



    
    
    

