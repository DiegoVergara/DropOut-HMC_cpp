'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import pandas as pd
import numpy as np

X_train = pd.read_csv("../data/birds/X_train.csv", sep =",", names = None, header = None)
Y_train = pd.read_csv("../data/birds/Y_train.csv", sep =",", names = None, header = None)

nb_classes = len(Y_train[0].unique())

X_test = pd.read_csv("../data/birds/X_test.csv", sep =",", names = None, header = None)
Y_test = pd.read_csv("../data/birds/Y_test.csv", sep =",", names = None, header = None)

assert(len(X_train) == len(Y_train))
assert(len(X_test) == len(Y_test))

print("Features Shape: {}".format(X_train.shape[1]))
print("Training Set:   {} samples".format(X_train.shape))
print("Test Set:       {} samples".format(X_test.shape))

dim = X_train.shape[1]
rows = X_train.shape[0]

# Parameters
learning_rate = 0.01
training_epochs = 35
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, nb_classes])

# Set model weights
W = tf.Variable(tf.zeros([dim, nb_classes]))
b = tf.Variable(tf.zeros([nb_classes]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def next_batch(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(rows/batch_size)
        # Loop over all batches
        for i,j in zip(next_batch(X_train,batch_size),next_batch(Y_train,batch_size)):
            batch_xs, batch_ys = i.values, indices_to_one_hot(j, nb_classes)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(indices_to_one_hot(Y_test, nb_classes), 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: X_test, y: indices_to_one_hot(Y_test, nb_classes)}))