# Backpropagation Dense Layer Demo for CS 675 at UMass Boston
# Network with One Sigmoidal Hidden Layer and a Softmax Output Layer

import matplotlib.pyplot as plt
import numpy as np

# Each training exemplar consists of an input vector x and a scalar output y that indicates class membership

# Two-class example
# x_train = np.array([[-0.44, 0.53], [-0.34, 0.47], [0.51, -0.34], [0.43, -0.48], [-0.32, -0.61], [-0.25, -0.41], [0.32, 0.21], [0.39, 0.43]])
# y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Three-class example (easy)
# x_train = np.array([[-0.62, 0.13], [-0.54, -0.27], [-0.55, 0.11], [0.73, 0.51], [0.63, 0.49], [0.46, -0.47], [0.52, -0.61], [0.39, -0.72]])
# y_train = np.array([0, 0, 0, 1, 1, 2, 2, 2])

# Three-class example (difficult)
x_train = np.array([[-0.62, 0.13], [-0.54, -0.27], [0.55, 0.21], [0.61, -0.09], 
                    [0.73, 0.51], [0.63, 0.49], [0.02, 0.04], [0.11, -0.08],
                    [0.46, -0.47], [0.52, -0.61], [-0.39, 0.72], [-0.46, 0.53]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

num_epochs = 2000
display_interval = 50   # Number of epochs between display updates
eta = 0.7               # Learning rate
decay_factor = 0.999    # Multiply eta with this value after each epoch

size_input = x_train.shape[1]           # Number of input-layer neurons must match size of the input vectors
size_hidden = 10                        # Number of hidden-layer neurons can be freely chosen
size_output = len(np.unique(y_train))   # Number of output-layer neurons must match number of classes

# Weight matrices for hidden and output layers. Each row contains the weights for one neuron, inluding its bias as the last (rightmost) element  
weights_h = np.random.normal(loc=0.0, scale=0.2, size=(size_hidden, size_input + 1))
weights_o = np.random.normal(loc=0.0, scale=0.2, size=(size_output, size_hidden + 1))

x_grid = np.array([[(x1 - 50)/50, (x2 - 50)/50] for x1 in range(100) for x2 in range(100)])     # a 2D grid used for visualizing the network function

# Compute the activations of a layer of sigmoidal neurons given its input x (vector of activations in previous layer)
# and its weight matrix w. The output is a vector containing the activation of each neuron in the current layer.   
def sigmoidal_layer_activation(x, w):
    pre_activation = np.matmul(w, x)
    return 1.0/(1.0 + np.exp(-pre_activation))

# Compute the activations of a softmax layer (same input/output format as for the sigmoidal layer) 
def softmax_layer_activation(x, w):
    pre_activation = np.matmul(w, x)
    exp_activation = np.exp(pre_activation)
    return exp_activation / np.sum(exp_activation)

# Compute activations of all neurons in the network and the classification result for input x
def feedforward_results(x, w_h, w_o):
    activation_h = sigmoidal_layer_activation(np.append(x, 1.0), w_h)           # We need to append the constant input of 1 for the bias weights
    activation_o = softmax_layer_activation(np.append(activation_h, 1.0), w_o)
    class_index = np.argmax(activation_o)                                       # The neuron with the greates activation determines class membership 
    return activation_h, activation_o, class_index
    
def show_network_function(x_list, y_list, weights_h, weights_o, epoch):
    fig = plt.figure(1, figsize=(6,6), dpi=100)
    plt.clf()

    marker_colors = np.array([(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)])  # Class 0: red; Class 1: blue, Class 2: green
    region_colors = np.array([(1.0, 0.5, 0.5), (0.5, 0.5, 1.0), (0.5, 1.0, 0.5)])

    y_grid = np.array([feedforward_results(x, weights_h, weights_o)[2] for x in x_grid])

    # Show the network's current discriminant (classification) function
    plt.scatter(x_grid[:, 0], x_grid[:, 1], s=50, c=region_colors[y_grid])
    
    # Show the training exemplars
    for cl in np.unique(y_list):
        plt.plot(x_list[y_list == cl, 0], x_list[y_list == cl, 1], 'o', markersize=8, markerfacecolor=marker_colors[cl], markeredgecolor=[0, 0, 0], label='Class ' + str(cl))
    
    plt.title('Network Function after Epoch %d'%(epoch))
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])    
    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(0.01)
    plt.waitforbuttonpress()

num_train_exemplars = x_train.shape[0] 

for epoch in range(num_epochs + 1):
    update_display = (epoch % display_interval == 0)
    num_correct = 0
    loss_sum = 0.0
    train_order = list(range(num_train_exemplars))
    np.random.shuffle(train_order)      # Randomize order in which exemplars are picked for training in each epoch
    
    for i in train_order:
        activation_h, activation_o, class_index = feedforward_results(x_train[i], weights_h, weights_o)
        y = np.zeros(size_output)
        y[y_train[i]] = 1.0         # The desired (ideal) output for an exemplar of class n is 1 for output neuron n and 0 for all other output neurons 
       
        if class_index == y_train[i]:
            num_correct += 1
        
        loss_sum -= np.log(activation_o[y_train[i]])    # Add cross-entropy loss for current exemplar to the loss sum 
     
        # Error (loss) gradient with regard to the pre-activations in the output layer (vector of length size_output)
        dE_dPre_o = activation_o - y
        
        # Loss gradient with regard to the output layer weights (matrix of same size as weights_o)
        # Note that we expand dE_dPre_o to make it a column vector, and we append a value of 1 to activation_h to represent the bias input
        dE_dWeights_o = np.expand_dims(dE_dPre_o, 1) * np.append(activation_h, 1.0)
       
        # Loss gradient with regard to the activations in the hidden layer (vector of length size_hidden)
        # Here, we disregard the bias weights in weights_o because they are irrelevant to this gradient  
        dE_dAct_h = np.sum(weights_o.T[:-1] * dE_dPre_o, axis=1)

        # Loss gradient with regard to the pre-activations in the hidden layer (vector of length size_hidden)
        # It is computed by multiplying dE_dAct_h with the gradient of hidden layer activations with regard to their pre-activations (derivative of sigmoid function)
        dE_dPre_h = dE_dAct_h * activation_h * (1.0 - activation_h)
        
        # Loss gradient with regard to the hidden layer weights (matrix of same size as weights_h)
        # Note that we expand dE_dPre_h to make it a column vector, and we append a value of 1 to the input vectors to represent the bias input
        dE_dWeights_h = np.expand_dims(dE_dPre_h, 1) * np.append(x_train[i], 1.0)

        # Update all weights according to the gradient descent rule 
        weights_h -= eta * dE_dWeights_h
        weights_o -= eta * dE_dWeights_o
        
    if update_display:
        loss = loss_sum / num_train_exemplars
        accuracy = num_correct / num_train_exemplars 
        print('Epoch %d: eta = %.4f, Loss = %.4f, accuracy = %.2f percent'%(epoch, eta, loss, 100.0 * accuracy))
        show_network_function(x_train, y_train, weights_h, weights_o, epoch)

    eta *= decay_factor

    
    
    

