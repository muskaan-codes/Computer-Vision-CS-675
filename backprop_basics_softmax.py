# Basic Backpropagation Demo for CS 675 at UMass Boston
# Network with 2 Input Units, 2 Sigmoidal Hidden Units, and 2 Softmax Output Units

import matplotlib.pyplot as plt
import numpy as np
import math

# Each training exemplar consists of a two-dimensional input x and a scalar output y that indicates class membership (0 or 1)
# The corresponding elements of x_train and y_train define 8 exemplars of this type.    
x_train = np.array([[-0.44, 0.53], [-0.34, 0.47], [0.51, -0.34], [0.43, -0.48], [-0.32, -0.61], [-0.25, -0.41], [0.32, 0.21], [0.39, 0.43]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

num_epochs = 300
display_interval = 10   # Number of epochs between display updates
eta = 0.7               # Learning rate
decay_factor = 0.997    # Multiply eta with this value after each epoch

# Weight vectors for the four computational neurons in the network, whose last vector element serves as a bias. 
# Here we initialize the weight vectors with specific values, but usually this is done using random numbers. 
weights_h0 = np.array([0.2, -0.5, -0.1])     # Weights for hidden-layer unit h0
weights_h1 = np.array([-0.3, 0.1, -0.2])     # Weights for hidden-layer unit h1
weights_o0 = np.array([0.5, 0.5, 0.1])       # Weights for output-layer unit o0
weights_o1 = np.array([0.2, -0.3, 0.2])      # Weights for output-layer unit o1

x_grid = np.array([[(x1 - 50)/50, (x2 - 50)/50] for x1 in range(100) for x2 in range(100)])     # a 2D grid used for visualizing the network function

def sigmoid_neuron_output(x, w):
    pre_activation = x[0] * w[0] + x[1] * w[1] + w[2]   # w[2] is a bias, i.e., a weight for a constant input of 1. 
    return 1.0/(1.0 + math.exp(-pre_activation))

def softmax_neuron_output(x, w):
    pre_activation = x[0] * w[0] + x[1] * w[1] + w[2]   # w[2] is a bias, i.e., a weight for a constant input of 1. 
    return math.exp(pre_activation)

def network_output(x, w_h0, w_h1, w_o0, w_o1):
    output_h0 = sigmoid_neuron_output(x, w_h0)
    output_h1 = sigmoid_neuron_output(x, w_h1)
    output_o0 = softmax_neuron_output([output_h0, output_h1], w_o0)
    output_o1 = softmax_neuron_output([output_h0, output_h1], w_o1)
    network_output_sum = output_o0 + output_o1
    output_o0 /= network_output_sum
    output_o1 /= network_output_sum
    if output_o0 > output_o1:
        classification_result = 0
    else:
        classification_result = 1
    return output_h0, output_h1, output_o0, output_o1, classification_result
    
def show_network_function(x_list, y_list, weights_h0, weights_h1, weights_o0, weights_o1, epoch):
    fig = plt.figure(1, figsize=(8,6), dpi=100)
    plt.clf()

    marker_colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]  # Class 0: red; Class 1: blue

    y_grid = np.array([network_output(x, weights_h0, weights_h1, weights_o0, weights_o1)[3] for x in x_grid])

    # Show the network's current discriminant (classification) function
    plt.scatter(x_grid[:, 0], x_grid[:, 1], s=50, c=y_grid, cmap='RdYlBu')
    
    # Show the training exemplars
    for cl in [0, 1]:
        plt.plot(x_list[y_list == cl, 0], x_list[y_list == cl, 1], 'o', markersize=8, markerfacecolor=marker_colors[cl], markeredgecolor=[0, 0, 0], label='Class ' + str(cl))
    
    plt.title('Network Function after Epoch %d'%(epoch))
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])    
    plt.legend(loc='upper right')
    plt.colorbar(location='right')
    plt.draw()
    plt.pause(0.01)
    plt.waitforbuttonpress()

num_train_exemplars = x_train.shape[0] 

for epoch in range(num_epochs + 1):
    update_display = (epoch % display_interval == 0)
    num_correct = 0
    loss_sum = 0.0
    train_order = list(range(num_train_exemplars))
    np.random.shuffle(train_order)
    
    for i in train_order:
        output_h0, output_h1, output_o0, output_o1, classification_result = network_output(x_train[i], weights_h0, weights_h1, weights_o0, weights_o1)
        if y_train[i] == 0:
            output_y0, output_y1 = 1, 0
        else:
            output_y0, output_y1 = 0, 1

        eval_string = 'Mistake!'
        if classification_result == y_train[i]:
            num_correct += 1
            eval_string = 'Correct!'
        
        if update_display:
            print('Exemplar %d: x = [%.3f, %.3f], y = [%.3f, %.3f] (Class %d), o = [%.3f, %.3f] (Class %d) -> %s'\
                %(i + 1, x_train[i, 0], x_train[i, 1], output_y0, output_y1, y_train[i], output_o0, output_o1, classification_result, eval_string))

        loss_sum -= output_y0 * math.log(output_o0) + output_y1 * math.log(output_o1)
    
        dE_weights_h0 = np.zeros(3)
        dE_weights_h1 = np.zeros(3)
        dE_weights_o0 = np.zeros(3)
        dE_weights_o1 = np.zeros(3)
        
        dE_dPre_o0 = output_o0 - output_y0
        dE_dPre_o1 = output_o1 - output_y1

        dE_weights_o0[0] = output_h0 * dE_dPre_o0
        dE_weights_o0[1] = output_h1 * dE_dPre_o0
        dE_weights_o0[2] =         1 * dE_dPre_o0
        
        dE_weights_o1[0] = output_h0 * dE_dPre_o1
        dE_weights_o1[1] = output_h1 * dE_dPre_o1
        dE_weights_o1[2] =         1 * dE_dPre_o1

        dE_dPost_h0 = weights_o0[0] * dE_dPre_o0 + weights_o1[0] * dE_dPre_o1 
        dE_dPost_h1 = weights_o0[1] * dE_dPre_o0 + weights_o1[1] * dE_dPre_o1 
        
        dE_dPre_h0 = dE_dPost_h0 * output_h0 * (1.0 - output_h0)
        dE_dPre_h1 = dE_dPost_h1 * output_h1 * (1.0 - output_h1)
        
        dE_weights_h0[0] = x_train[i, 0] * dE_dPre_h0
        dE_weights_h0[1] = x_train[i, 1] * dE_dPre_h0
        dE_weights_h0[2] =             1 * dE_dPre_h0
        
        dE_weights_h1[0] = x_train[i, 0] * dE_dPre_h1
        dE_weights_h1[1] = x_train[i, 1] * dE_dPre_h1
        dE_weights_h1[2] =             1 * dE_dPre_h1
        
        weights_h0 -= eta * dE_weights_h0
        weights_h1 -= eta * dE_weights_h1
        weights_o0 -= eta * dE_weights_o0
        weights_o1 -= eta * dE_weights_o1

    if update_display:
        loss = loss_sum / num_train_exemplars
        accuracy = num_correct / num_train_exemplars 
        print('\nEpoch %d: eta = %.4f, loss = %.4f, accuracy = %.2f percent'%(epoch, eta, loss, 100.0 * accuracy))
        show_network_function(x_train, y_train, weights_h0, weights_h1, weights_o0, weights_o1, epoch)

    eta *= decay_factor
    
    

