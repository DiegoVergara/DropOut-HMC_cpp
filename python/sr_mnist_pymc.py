import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
from scipy import optimize
from pymc3 import geweke,traceplot
from pymc3 import sample_ppc
from theano import shared

X_train = pd.read_csv("../data/MNIST/X_train.csv", sep =",", names = None, header = None)
Y_train = pd.read_csv("../data/MNIST/Y_train.csv", sep =",", names = None, header = None)
X_test = pd.read_csv("../data/MNIST/X_test.csv", sep =",", names = None, header = None)
Y_test = pd.read_csv("../data/MNIST/Y_test.csv", sep =",", names = None, header = None)
X_train = X_train - X_train.mean(axis=0);
X_test = X_test - X_train.mean(axis=0);

#X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
#X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

#X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0)- X_train.min(axis=0))
#X_test = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0)- X_train.min(axis=0))

n_iter = 100
step_size = 0.1
dim = X_train.shape[1]
classes = len(Y_train[0].unique())
print("model:")
with pm.Model() as iris_model:
	sd = pm.Gamma('sd', alpha=10,beta=1)
	alfa = pm.Normal('alfa', mu=0, sd=sd, shape=classes)
	beta = pm.Normal('beta', mu=0, sd=sd, shape=(dim,classes))
	mu = alfa + pm.math.dot(X_train, beta)
	theta = pm.Deterministic('theta', tt.nnet.softmax(mu))
	yl = pm.Categorical('yl', p=theta, observed=Y_train)
	step = pm.HamiltonianMC(step_scale=step_size,path_length=1.0,is_cov=True)
	iris_trace = pm.sample(n_iter,step)

print("post_pred:")

with iris_model:
    post_pred = pm.sample_ppc(iris_trace, samples=10)


traceplot(iris_trace)
plt.show()
plt.close()