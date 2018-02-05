import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal, Empirical
import edward as ed
import pandas as pd
from sklearn.utils import shuffle

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

#X_train, Y_train = shuffle(X_train, Y_train)

#X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
#X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

#X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0)- X_train.min(axis=0))
#X_test = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0)- X_train.min(axis=0))

dim = X_train.shape[1]
rows = X_train.shape[0]


sns.set(color_codes=True)
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) 
ed.set_seed(314159)
N = 5   # number of images in a minibatch.
D = dim   # number of features.
K = nb_classes    # number of classes.
n_samples=1000

x = tf.placeholder(tf.float32, [None, D])
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
y = Categorical(tf.matmul(tf.cast(x, tf.float32),w)+b)


#qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K])))) 
#qb = Normal(loc=tf.Variable(tf.random_normal([K])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

qw= Empirical(params=tf.Variable(tf.random_normal([n_samples,D,K])))
qb= Empirical(params=tf.Variable(tf.random_normal([n_samples,K])))
        
y_ph = tf.placeholder(tf.int32, [N])
#inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})
inference = ed.SGHMC({w: qw, b: qb}, data={y:y_ph})


inference.initialize(n_iter=n_samples, n_print=50, scale={y: float(rows) / N},step_size=0.1,friction=1.0)
#inference.initialize(n_iter=n_samples, n_print=50,step_size=0.1,friction=2.0)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#X_train=mnist.train.images

def next_batch(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


for i,j in zip(next_batch(X_train,N),next_batch(Y_train,N)):
    X_batch, Y_batch = i.values, j.values.flatten()
    #print(X_batch.shape)
    #X_batch = X_batch - X_train.mean(axis=0)
    #Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)
#print("Run ok")

'''
X_test = mnist.test.images
X_test = X_test - X_train.mean(axis=0);
Y_test = np.argmax(mnist.test.labels,axis=1)
'''

n_samples = 100
prob_lst = []
samples = []
w_samples = []
b_samples = []
for _ in range(n_samples):
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test.values, tf.float32),w_samp ) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp,[-1]),b_samp],0)
    samples.append(sample.eval())


accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test.values.flatten()).mean()*100
    accy_test.append(acc)

#sns.distplot(accy_test)

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the BIRDS test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.savefig("BIRDS_t_data_acc_freq.png")
#plt.show()
plt.close()


Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1)
print("accuracy in predicting the test data = ", (Y_pred == Y_test.values.flatten()).mean()*100)

'''
samples_df = pd.DataFrame(data = samples, index=range(n_samples))
samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values,columns=["W_0", "W_1", "W_2", "W_3", "W_4"])
g = sns.PairGrid(samples_5, diag_sharey=False)
g.map_lower(sns.kdeplot, n_levels = 4,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,legend=False)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Joint posterior distribution of the first 5 weights')
plt.savefig("BIRDS_first_5_w.png")
plt.show()
plt.close()
'''
'''
test_image = X_test[0:1]
test_label = Y_test[0]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("BIRDS_gt.png")
plt.show()
plt.close()
'''

D = 200
sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test[280:281], tf.float32),w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

print(Y_test[280:281])
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(D))
plt.xticks(np.arange(53,68))
plt.xlim(53,68)
plt.xlabel("Accuracy of the prediction of the test")
plt.ylabel("Frequency")
plt.savefig("BIRDS_acc_freq28.png")
#plt.show()
plt.close()


sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test[620:621], tf.float32),w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

print(Y_test[620:621])
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(D))
plt.xticks(np.arange(53,68))
plt.xlim(53,68)
plt.xlabel("Accuracy of the prediction of the test")
plt.ylabel("Frequency")
plt.savefig("BIRDS_acc_freq62.png")
#plt.show()
plt.close()

sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test[647:648], tf.float32),w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

print(Y_test[647:648])
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(D))
plt.xticks(np.arange(53,68))
plt.xlim(53,68)
plt.xlabel("Accuracy of the prediction of the test")
plt.ylabel("Frequency")
plt.savefig("BIRDS_acc_freq64.png")
#plt.show()
plt.close()

sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test[683:684], tf.float32),w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

print(Y_test[683:684])
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(D))
plt.xticks(np.arange(53,68))
plt.xlim(53,68)
plt.xlabel("Accuracy of the prediction of the test")
plt.ylabel("Frequency")
plt.savefig("BIRDS_acc_freq68.png")
#plt.show()
plt.close()

sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test[746:747], tf.float32),w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

print(Y_test[746:747])
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(D))
plt.xticks(np.arange(53,68))
plt.xlim(53,68)
plt.xlabel("Accuracy of the prediction of the test")
plt.ylabel("Frequency")
plt.savefig("BIRDS_acc_freq74.png")
#plt.show()
plt.close()

sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(X_test[830:831], tf.float32),w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

print(Y_test[830:831])
plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(D))
plt.xticks(np.arange(53,68))
plt.xlim(53,68)
plt.xlabel("Accuracy of the prediction of the test")
plt.ylabel("Frequency")
plt.savefig("BIRDS_acc_freq83.png")
#plt.show()
plt.close()
'''
git clone git@github.com:davidflanagan/notMNIST-to-MNIST.git
mkdir notMNIST_data
cp notMNIST-to-MNIST/*.gz notMNIST_data
'''
'''
not_mnist = input_data.read_data_sets("./notMNIST_data/", one_hot=True) 


Xnm_test = not_mnist.test.images
Ynm_test = np.argmax(not_mnist.test.labels,axis=1)


test_image = Xnm_test[0]
test_label = Ynm_test[0]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("gt2.png")
plt.show()
plt.close()

nm_sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul(tf.cast(Xnm_test[0:1], tf.float32),w_samp ) + b_samp)
    nm_sing_img_probs.append(prob.eval())


plt.hist(np.argmax(nm_sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Accuracy of the prediction of the test letter")
plt.ylabel("Frequency")
plt.savefig("letter_acc_freq.png")
plt.show()
plt.close()

'''
