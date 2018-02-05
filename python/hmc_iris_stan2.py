import pandas as pd
import numpy as np
import pystan 
from pystan import StanModel
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks",color_codes=True)

def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), 
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

'''
iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_0 = df[x_n].values
'''

palette = 'muted'
sns.set_palette(palette); sns.set_color_codes(palette)
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)

iris = sns.load_dataset("iris")
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_0 = df[x_n].values
x_0 = (x_0 - x_0.mean(axis=0)) / x_0.std(axis=0)

iris_data={'n':x_0.shape[0],
           'species':y_0,
           'sepal_length':x_0[:,0],
           'sepal_width':x_0[:,1]}


v_iter = [1100, 2000, 11000]
v_delta = [0.01, 0.05]
v_steps = [10, 100]

for n_iter in v_iter:
    for delta in v_delta:
        for n_steps in v_steps:
          print 'n_iter: '+str(n_iter)+', delta: '+str(delta)+', n_steps: '+str(n_steps)
          m = StanModel(file='iris.stan')
          control = {'stepsize': delta, 'int_time': n_steps*delta}
          fit=m.sampling(data=iris_data,iter=n_iter,chains=1,warmup=1000,algorithm='HMC', control = control)
          trace_0 = pd.DataFrame(fit.extract(['alpha','beta_0','beta_1']))
          print(fit.summary())
          varnames=['alpha','beta0', 'beta1']
          trace_0.columns = varnames
          g=sns.pairplot(trace_0[int((n_iter-1000)/10):n_iter-1000])
          g.savefig("stan_"+str(n_iter-1000)+"_"+str(delta)+"_"+str(n_steps)+".png")
          plt.close()
          del(g)
          del(trace_0)
          #sns.plt.show()

'''
trace ~ bernoulli(inv_logit(trace_0[0]+trace_0[1]*y_0))

naive_hpd(trace_0.mean(axis=0))
plt.show()
'''
