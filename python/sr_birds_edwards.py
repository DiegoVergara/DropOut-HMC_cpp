import tensorflow as tf
import edward as ed
from edward.models import Normal, Empirical,Categorical
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#device_count = {'GPU': 0}
X_train = pd.read_csv("../data/birds/X_train.csv", sep =",", names = None, header = None)
y_train = pd.read_csv("../data/birds/Y_train.csv", sep =",", names = None, header = None)
nb_classes = len(y_train[0].unique())

X_test = pd.read_csv("../data/birds/X_test.csv", sep =",", names = None, header = None)
y_test = pd.read_csv("../data/birds/Y_test.csv", sep =",", names = None, header = None)

assert(len(X_train) == len(y_train))
assert(len(X_test) == len(y_test))

print("Features Shape: {}".format(X_train.shape[1]))
print()
print("Training Set:   {} samples".format(X_train.shape))
print("Test Set:       {} samples".format(X_test.shape))

X_train, y_train = shuffle(X_train, y_train)

dim = X_train.shape[1]

EPOCHS = 10
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

def Softmax(xtrain,dic):    
    mu = 0
    sigma = 0.1
    f_W=dic['weights']
    f_b=dic['bias']
    logits = tf.matmul(tf.cast(xtrain, tf.float32), f_W) + f_b
    return tf.nn.softmax(logits)

with tf.name_scope("model"):
    weights=  Normal(loc=tf.ones([dim,nb_classes]),scale=tf.ones([dim,nb_classes]),name='weights')
    bias = Normal(loc=tf.zeros([nb_classes]),scale=tf.ones([nb_classes]),name='bias')
    dic={'weights':weights,'bias':bias}
    X= tf.placeholder(tf.float32, shape=[None, dim])
    y=tf.identity(Categorical(Softmax(X,dic)),name="y")

with tf.name_scope("posterior"):
    Nsamples=1000     
    with tf.name_scope("qweights"): 
        qweights = Empirical(params=tf.Variable(tf.random_normal([Nsamples,dim,nb_classes])))
     
    with tf.name_scope("qbias"):     
        qbias=Empirical(params=tf.Variable(tf.random_normal([Nsamples,nb_classes])))
        

N=100
x = tf.placeholder(tf.float32, shape=[None, dim])
y_ph = tf.placeholder(tf.int32, shape=[None])

inference = ed.SGHMC({weights:qweights,bias:qbias},data={y:y_ph})

inference.initialize(n_iter=1000, n_print=100,step_size=1e-1, friction=1.0)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print("Session")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    new_y_train = y_train.values.flatten()
    print("Training...")
    print(inference.n_iter)
    for i in range(inference.n_iter):
        info_dict=inference.update(feed_dict={x: X_train, y_ph: new_y_train})
        inference.print_progress(info_dict)
    #saver = tf.train.Saver()
    #saver.save(sess, './bayesianlenet')

print("Testing")
print("Test Set:       {} samples".format(len(X_test)))
with tf.name_scope("testing"):
    nTestSamples=30
    result=[]
    for i in range(nTestSamples):
        tweights=qweights.sample()
        tbias=qbias.sample()
        dic={'weights':tweights,'bias':tbias}
        ypred=Categorical(Softmax(X_test,dic))
        print(ypred.eval())
        print(y_test.values.flatten())
        match = (ypred.eval()==y_test.values.flatten()).sum()
        print(match)
        print("Test Accuracy:       {} %".format(str(round(match*100/len(X_test),6))))

#import matplotlib.pyplot as plt
#plt.hist(np.array(result)/100)
#plt.show()

