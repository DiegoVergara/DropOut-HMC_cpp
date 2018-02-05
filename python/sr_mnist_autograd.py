"""A multi-layer perceptron for classification of MNIST handwritten digits."""
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam,sgd
from data import load_mnist


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    '''
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
    '''
    return [(scale * np.ones((m, n)),   # weight matrix
             scale * np.ones(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
    return outputs - logsumexp(outputs, axis=1, keepdims=True)

def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)

def log_posterior(params, inputs, targets, L2_reg):
    #n_data=inputs.shape[0]
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    #log_lik = np.mean(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik

def accuracy(params, inputs, targets):
    target_class    = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


if __name__ == '__main__':
    # Model parameters
    import pandas as pd
    
    '''
    dataset = "IRIS"
    train_images = pd.read_csv("../data/"+dataset+"/X_train.csv", sep =",", names = None, header = None).values
    train_labels = pd.read_csv("../data/"+dataset+"/Y_train.csv", sep =",", names = None, header = None).values
    test_images = pd.read_csv("../data/"+dataset+"/X_test.csv", sep =",", names = None, header = None).values
    test_labels = pd.read_csv("../data/"+dataset+"/Y_test.csv", sep =",", names = None, header = None).values

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.transform(test_labels)
    
    layer_sizes = [4,3]
    L2_reg = 10.0
    batch_size = 100
    step_size = 0.001
    '''

    '''
    data = pd.read_csv("../data/adience.csv", sep =",", names = None, header = None).values
    labels = pd.read_csv("../data/age_label.csv", sep =",", names = None, header = None).values

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    train_images = data[:14000, :]
    train_labels = labels[:14000, :]
    test_images = data[14000:, :]
    test_labels = labels[14000:, :]

    layer_sizes = [3776,6]
    L2_reg = 0.1
    batch_size = 1000
    step_size = 0.01   
    '''
    
    
    dataset = "MNIST"
    train_images = pd.read_csv("../data/"+dataset+"/X_train.csv", sep =",", names = None, header = None).values
    train_labels = pd.read_csv("../data/"+dataset+"/Y_train.csv", sep =",", names = None, header = None).values
    test_images = pd.read_csv("../data/"+dataset+"/X_test.csv", sep =",", names = None, header = None).values
    test_labels = pd.read_csv("../data/"+dataset+"/Y_test.csv", sep =",", names = None, header = None).values

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.transform(test_labels)
    
    layer_sizes = [784,10]
    L2_reg = 1.0
    batch_size = 1000
    step_size = 0.001
    

    # Training parameters
    param_scale = 0.1
    
    num_epochs = 100

    init_params = init_random_params(param_scale, layer_sizes)

    num_batches = int(np.ceil(len(train_images) / batch_size))
    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(params, train_images[idx], train_labels[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |     Loss        |   Train accuracy  |       Test accuracy  ")
    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, train_images, train_labels)
            test_acc  = accuracy(params, test_images, test_labels)
            #print(objective_grad(params,iter))
            #print("{:15}|{:20}|{:20}".format(iter//num_batches, train_acc, test_acc))
            if(iter%10==0): print("{:15}|{:20}|{:20}|{:20}".format(iter//num_batches, objective(params,iter),train_acc, test_acc))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = sgd(objective_grad, init_params, step_size=step_size,num_iters=num_epochs * num_batches, callback=print_perf)

