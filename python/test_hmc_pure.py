import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
import pymc3 as pm
from autograd import grad
from autograd.util import quick_grad_check
from autograd.util import flatten
from autograd.util import flatten_func
from builtins import range
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

iris = sns.load_dataset("iris")
#df = iris.query("species == ('virginica', 'versicolor')")
df = iris.query("species == ('setosa', 'versicolor')")
y_train = pd.Categorical(df['species']).codes
#x_n = ['petal_length', 'petal_width']
x_n = ['sepal_length', 'sepal_width']
x_train = df[x_n].values
x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)

alpha=100.

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def logistic_predictions(weights,bias, x_train):
    # Outputs probability of a label being true according to logistic model.
    eps=1e-15
    return sigmoid(np.dot(x_train, weights)+bias).clip(eps,1-eps)

def training_loss(params):
    # Training loss is the negative log-likelihood of the training labels.
    y_pred = logistic_predictions(params['weights'],params['bias'], x_train)
    dim=len(params['weights'])
    cov=alpha*np.eye(dim)
    log_alpha=-np.log(np.sqrt(2*np.pi))- np.log(np.sqrt(alpha))-0.5*(params['bias']**2)/(alpha)
    log_prior= -np.log(np.sqrt(2*np.pi)) -0.5 * np.log(np.linalg.det(cov)) - 0.5*np.dot(np.dot(params['weights'].T,(np.linalg.inv(cov))),(params['weights']))
    log_likelihood = y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)
    return np.sum(log_likelihood)+log_prior+log_alpha


# Build a function that returns gradients of training loss using autograd.

init_params={'weights':np.array(np.ones(x_train.shape[1])),'bias':1}
flattened_obj, unflatten, flattened_init_params =flatten_func(training_loss, init_params)
# Check the gradients numerically, just to be safe.

training_gradient_fun = grad(flattened_obj)

n_iter = 10000
warmup = 1000
delta = 0.01
path_length = 1.0
n_steps = int(path_length/delta)

import hamiltonian1 as hmc1
import hamiltonian2 as hmc2


print 'Descriptors: '+str(x_n)
print 'Params: n_iter: '+str(n_iter)+', warmup: '+str(warmup)+', delta: '+str(delta)+', n_steps: '+str(n_steps)
sampler0 = hmc1.Hamiltonian1(logp=flattened_obj, grad=training_gradient_fun, start=flattened_init_params, step_size=delta, path_length=path_length)
samples0 = sampler0.sample(warmup)
print('Warmup, acceptance : %f %%'%sampler0.acceptance_rate())
df = pd.DataFrame(samples0.tolist())
varnames=['alpha','beta0', 'beta1']
df.columns = varnames
g = sns.pairplot(df[int(warmup/10):warmup])
g.savefig("hmc_nocovar_"+str(n_iter)+"_"+str(delta)+"_"+str(n_steps)+".png")
plt.close()
del(g)
del(df)
cov = np.cov(samples0[int(warmup/10):warmup], rowvar = False)
#cov = np.diagflat(np.var(samples0, axis=0))

sampler = hmc2.Hamiltonian2(logp=flattened_obj, grad=training_gradient_fun, start=flattened_init_params, cov = cov, step_size=delta, path_length=path_length)
samples = sampler.sample(n_iter)
print('HMC, acceptance : %f %%'%sampler.acceptance_rate())
df0 = pd.DataFrame(samples.tolist())
df0.columns = varnames
g0 = sns.pairplot(df0[int(n_iter/10):n_iter])
g0.savefig("hmc_covar_"+str(n_iter)+"_"+str(delta)+"_"+str(n_steps)+".png")
plt.close()
del(g0)
del(df0)

'''
mean = unflatten(samples.mean(axis=0))

rows = int(samples.shape[0])
cols = int(len(x_train[:,0]))
bd = np.zeros([rows, cols])
for i in xrange(0,rows):
    bd[i,:] = -samples[i,0]/samples[i,2] -samples[i,1]/samples[i,2] * x_train[:,0]    

idx = np.argsort(x_train[:,0])
bd2 = bd.mean(axis=0)[idx]
plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=plt.cm.winter)
plt.plot(x_train[:,0][idx], bd2, color='r');
bd_hpd = pm.hpd(bd)[idx]
plt.fill_between(x_train[:,0][idx], bd_hpd[:,0], bd_hpd[:,1], color='r', alpha=0.5);
plt.xlabel(x_n[0], fontsize=12)
plt.ylabel(x_n[1], fontsize=12)
plt.gcf().subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig("bd_"+str(warmup)+"_"+str(n_iter)+".png")
plt.close()

def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), 
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

naive_hpd(samples[:,0])
plt.savefig("hpd_alpha_"+str(warmup)+"_"+str(n_iter)+".png")
plt.close()

naive_hpd(samples[:,1])
plt.savefig("hpd_beta0_"+str(warmup)+"_"+str(n_iter)+".png")
plt.close()

naive_hpd(samples[:,2])
plt.savefig("hpd_beta1_"+str(warmup)+"_"+str(n_iter)+".png")
plt.close()


y_pred=logistic_predictions(mean['weights'],mean['bias'],x_train)>0.5
print('Clasification Report:')
print(classification_report(y_train, y_pred, target_names=['setosa', 'versicolor']))
print('Confusion Matrix:')
print confusion_matrix(y_train, y_pred)
'''
