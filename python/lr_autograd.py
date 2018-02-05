"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam,sgd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [( (1.0/m)*np.ones([m, n]),   # weight matrix
             (1.0/m)*np.ones(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a logistic regression model for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs=np.tanh(outputs)
    return sigmoid(outputs)

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    #n_data=inputs.shape[0]
    log_prior = -L2_reg * l2_norm(params)
    phi=neural_net_predict(params, inputs)
    log_lik = np.sum( (targets*np.log(phi))+(1-targets)*np.log(1-phi))
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = targets
    predicted_class = neural_net_predict(params, inputs) > 0.5
    return np.mean(predicted_class == target_class)


if __name__ == '__main__':
    # Model parameters
    import pandas as pd
    scaler = MinMaxScaler()
    dataset = 'IRIS'
    train_data = pd.read_csv("../data/"+dataset+"/X_train.csv", sep =",", names = None, header = None).values
    train_labels = pd.read_csv("../data/"+dataset+"/Y_train.csv", sep =",", names = None, header = None).values
    test_data = pd.read_csv("../data/"+dataset+"/X_test.csv", sep =",", names = None, header = None).values
    test_labels = pd.read_csv("../data/"+dataset+"/Y_test.csv", sep =",", names = None, header = None).values
    #scaler.fit(train_data)
    #train_data = scaler.transform(train_data)
    #test_data = scaler.transform(test_data)
    train_labels[train_labels==-1]=0
    layer_sizes = [train_data.shape[1],1]
    L2_reg = 0.001
    #L2_reg = 0.0
    batch_size = 100
    step_size = 0.1
    
    # Training parameters
    param_scale = 1.0
    
    num_epochs = 10

    init_params = init_params(param_scale, layer_sizes)

    num_batches = int(np.ceil(len(train_data) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, train_data[idx], train_labels[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |     Loss        |   Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_data, train_labels)
            test_acc  = accuracy(params, test_data, test_labels)
            #print(flatten(params))
            #print(objective_grad(params,iter))
            print("{:15}|{:20}|{:20}|{:20}".format(iter//num_batches, objective(params,iter),train_acc, test_acc))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = sgd(objective_grad, init_params, step_size=step_size,num_iters=num_epochs * num_batches, callback=print_perf)

