import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal, Empirical
import edward as ed
import pandas as pd

sns.set(color_codes=True)
mnist = input_data.read_data_sets("../data/MNIST/", one_hot=True) 
ed.set_seed(314159)
N = 1000   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.
n_samples=1000
epoch = 1000
friction=0.001
step_size = 1e-4

x = tf.placeholder(tf.float32, [None, D])
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
y = Categorical(tf.matmul(x,w)+b)


#qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K])))) 
#qb = Normal(loc=tf.Variable(tf.random_normal([K])),scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

qw= Empirical(params=tf.Variable(tf.random_normal([n_samples,D,K])))
qb= Empirical(params=tf.Variable(tf.random_normal([n_samples,K])))
        
y_ph = tf.placeholder(tf.int32, [N])
#inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})
inference = ed.SGHMC({w: qw, b: qb}, data={y:y_ph})


num_batches = float(mnist.train.num_examples) / N
inference.initialize(n_iter=epoch*num_batches, n_print=n_samples, scale={y: num_batches},step_size=step_size,friction=friction)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
X_train=mnist.train.images

for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    X_batch = X_batch - X_train.mean(axis=0);
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)

X_test = mnist.test.images
X_test = X_test - X_train.mean(axis=0);
Y_test = np.argmax(mnist.test.labels,axis=1)


n_samples = 10
prob_lst = []
samples = []
w_samples = []
b_samples = []
for _ in range(n_samples):
    w_samp = qw.sample()
    b_samp = qb.sample()
    w_samples.append(w_samp)
    b_samples.append(b_samp)
    prob = tf.nn.softmax(tf.matmul( X_test,w_samp ) + b_samp)
    prob_lst.append(prob.eval())
    sample = tf.concat([tf.reshape(w_samp,[-1]),b_samp],0)
    samples.append(sample.eval())


accy_test = []
for prob in prob_lst:
    y_trn_prd = np.argmax(prob,axis=1).astype(np.float32)
    acc = (y_trn_prd == Y_test).mean()*100
    accy_test.append(acc)

#sns.distplot(accy_test)
'''
plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.savefig("MNIST_t_data_acc_freq.png")
plt.show()
plt.close()
'''

prob_mean = np.mean(prob_lst,axis=0)
prob_std = np.std(prob_lst,axis=0)
prob_max = np.max(prob_mean,axis=1)


Y_pred = np.argmax(np.mean(prob_lst,axis=0),axis=1)
print("accuracy in predicting the test data = ", (Y_pred == Y_test).mean()*100)

result = np.concatenate((prob_mean, np.reshape(prob_max,(-1,1)), np.reshape(Y_pred,(-1,1)),np.reshape(Y_test,(-1,1)),prob_std),axis=1)
np.savetxt("mnist_analysis.csv", result, fmt="%1.3f", header ="mean_0, mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, mean_8, mean_9, max_prob, pred, GT, std_0, std_1, std_2, std_3, std_4, std_5, std_6, std_7, std_8, std_9",delimiter = ",")


'''
samples_df = pd.DataFrame(data = samples, index=range(n_samples))
samples_5 = pd.DataFrame(data = samples_df[list(range(5))].values,columns=["W_0", "W_1", "W_2", "W_3", "W_4"])
g = sns.PairGrid(samples_5, diag_sharey=False)
g.map_lower(sns.kdeplot, n_levels = 4,cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot,legend=False)
plt.subplots_adjust(top=0.95)
g.fig.suptitle('Joint posterior distribution of the first 5 weights')
plt.savefig("MNIST_first_5_w.png")
plt.show()
plt.close()
'''
'''
test_image = X_test[0:1]
test_label = Y_test[0]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("MNIST_gt.png")
plt.show()
plt.close()
'''

'''
sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul( X_test[0:1],w_samp ) + b_samp)
    sing_img_probs.append(prob.eval())

plt.hist(np.argmax(sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Accuracy of the prediction of the test digit")
plt.ylabel("Frequency")
plt.savefig("MNIST_digit_acc_freq.png")
plt.show()
plt.close()
'''

'''
git clone git@github.com:davidflanagan/notMNIST-to-MNIST.git
mkdir notMNIST_data
cp notMNIST-to-MNIST/*.gz notMNIST_data
'''
'''
not_mnist = input_data.read_data_sets("../data/notMNIST/", one_hot=True) 


Xnm_test = not_mnist.test.images
Ynm_test = np.argmax(not_mnist.test.labels,axis=1)


test_image = Xnm_test[0]
test_label = Ynm_test[0]
print('truth = ',test_label)
pixels = test_image.reshape((28, 28))
plt.imshow(pixels,cmap='Blues')
plt.savefig("NoMNIST_gt2.png")
plt.show()
plt.close()

nm_sing_img_probs = []
for w_samp,b_samp in zip(w_samples,b_samples):
    prob = tf.nn.softmax(tf.matmul( Xnm_test[0:1],w_samp ) + b_samp)
    nm_sing_img_probs.append(prob.eval())


plt.hist(np.argmax(nm_sing_img_probs,axis=2),bins=range(10))
plt.xticks(np.arange(0,10))
plt.xlim(0,10)
plt.xlabel("Accuracy of the prediction of the test letter")
plt.ylabel("Frequency")
plt.savefig("NoMNIST_letter_acc_freq.png")
plt.show()
plt.close()
'''
