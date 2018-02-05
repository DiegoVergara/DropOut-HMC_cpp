# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#from sklearn import datasets
import pymc3 as pm
import numpy as np
import pandas as pd
#from pymc3 import geweke
#import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
#import theano.tensor as tt
#from scipy.special import expit

def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), 
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

varnames=['alpha','beta0', 'beta1']
df_cplus = pd.read_csv("../build/hmc_weights.csv", names=varnames)
#df_cplus = pd.read_csv("../build/hmc_iris_warmup.csv")


print("HMC c++")

df_cplus = df_cplus[int(df_cplus.shape[0]/10):df_cplus.shape[0]]
#print(df_cplus.mean(axis = 0))
df_cplus.shape
#sns.set(style="ticks", color_codes=True)

g2 = sns.pairplot(df_cplus)
plt.suptitle('C++ HMC Distribution')
g2.savefig("cpp.png")
plt.show()
plt.close()

autocorrelation_plot(df_cplus['alpha'])
autocorrelation_plot(df_cplus['beta0'])
autocorrelation_plot(df_cplus['beta1'])
#plt.savefig("cpp_auto.png")
plt.show()
plt.close()

z0 = geweke(df_cplus['alpha'], intervals=15)

plt.plot(z0,'o')
#plt.hlines([-2, 2], 0, 15, linestyles='dotted')
#plt.xlim(0, 14)
#plt.ylim(-4, 4)
#plt.show()
#plt.close()

z1 = geweke(df_cplus['beta0'], intervals=15)

plt.plot(z1,'o')
#plt.hlines([-2, 2], 0, 15, linestyles='dotted')
#plt.xlim(0, 14)
#plt.ylim(-4, 4)
#plt.show()
#plt.close()

z2 = geweke(df_cplus['beta1'], intervals=15)

plt.plot(z2,'o')
plt.hlines([-2, 2], 0, 15, linestyles='dotted')
plt.xlim(0, 14)
plt.ylim(-4, 4)
plt.show()
'''
post1 = df_cplus.values
post1 = post1[:,0].T
naive_hpd(post1)
plt.show()
'''

