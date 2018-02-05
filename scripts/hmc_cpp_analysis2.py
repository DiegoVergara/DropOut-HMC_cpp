# -*- coding: utf-8 -*-
from __future__ import unicode_literals
#from sklearn import datasets
#import pymc3 as pm
import numpy as np
import pandas as pd
#from pymc3 import geweke
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot


'''
def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), 
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])
'''
def rhot(x, t):
    n = len(x)
    return np.corrcoef(x[0:(n-t)], x[t:n])[0,1]


def Geweke(trace, intervals, length):
    nsl=length
    # Divide the length of the trace (minus the burnin portion) into equal segments.
    # There will be two sets of n intervals, one for the early portion and one for
    # the late portion of the trace
    jump = int(0.9*len(trace)/(2*intervals))
    # Get rid of the burnin segment
    first = 0.1*len(trace)
    
    z =np.empty(intervals)
    # Loop through each pair of an interval
    for k in np.arange(0, intervals):
        # Define the start position of the current segment in the early portion
        baga = np.int(first+k*jump)
        # Define the start position of the current segment in the late portion
        # (starting at 50% of the trace)
        bagb = np.int(len(trace)/2 + k*jump)
        
        # Obtain the segments of length nsl for both portions
        sub_trace_a = np.array(trace[baga:baga+nsl])
        sub_trace_b = np.array(trace[bagb:bagb+nsl])
        
        # Compute the means of each segment
        theta_a = np.mean(sub_trace_a)
        theta_b = np.mean(sub_trace_b)
        rho_a, rho_b = 1.0, 1.0
        
        # Compute correlation correction factor
        for i in np.arange(int(0.1*nsl)):
            # Increments to measure correlation between groups of elements
            # in each subtrace
            rho_a += 2*rhot(sub_trace_a, i+1)
            rho_b += 2*rhot(sub_trace_b, i+1)
            
        # Adjust the standard deviations for correlation.
        var_a  = np.var(sub_trace_a)*rho_a/length
        var_b  = np.var(sub_trace_b)*rho_b/length
        
        # Estimate the z-score, an estimate of the differencee in means
        # divided by the standard deviation of the distribution of mean differences.
        z[k] = (theta_a-theta_b)/np.sqrt( var_a + var_b)
    
   	return z


df_cplus = pd.read_csv("hmc_posterior.csv")

print("HMC c++")

'''
df_cplus = df_cplus[int(df_cplus.shape[0]/10):df_cplus.shape[0]]
#print(df_cplus.mean(axis = 0))
df_cplus.shape
#sns.set(style="ticks", color_codes=True)

g2 = sns.pairplot(df_cplus)
plt.suptitle('C++ HMC Distribution')
g2.savefig("cpp.png")
plt.show()
plt.close()
'''
autocorrelation_plot(df_cplus.ix[:,0])
plt.show()
plt.close()
autocorrelation_plot(df_cplus.ix[:,1])
plt.show()
plt.close()
autocorrelation_plot(df_cplus.ix[:,2])
#plt.savefig("cpp_auto.png")
plt.show()
plt.close()

z1 = Geweke(df_cplus.ix[:,0], intervals=10, length=100)
#z1 = geweke(df_cplus.ix[:,0], intervals=10)
plt.plot(z1,'o')
plt.hlines([-2, 2], 0, 10, linestyles='dotted')
#plt.xlim(0, 14)
plt.ylim(-4, 4)
plt.show()
plt.close()

z2 = Geweke(df_cplus.ix[:,1], intervals=10, length=100)
#z2 = geweke(df_cplus.ix[:,1], intervals=10)
plt.plot(z2,'o')
plt.hlines([-2, 2], 0, 10, linestyles='dotted')
#plt.xlim(0, 14)
plt.ylim(-4, 4)
plt.show()
plt.close()

z3 = Geweke(df_cplus.ix[:,2], intervals=10, length=100)
#z3 = geweke(df_cplus.ix[:,2], intervals=10)
plt.plot(z3,'o')
plt.hlines([-2, 2], 0, 10, linestyles='dotted')
#plt.xlim(0, 14)
plt.ylim(-4, 4)
plt.show()
plt.close()


z3 = Geweke(df_cplus.ix[:,600], intervals=10, length=100)

plt.plot(z3,'o')
plt.hlines([-2, 2], 0, 10, linestyles='dotted')
#plt.xlim(0, 14)
plt.ylim(-4, 4)
plt.show()
plt.close()


'''
post1 = df_cplus.values
post1 = post1[:,0].T
naive_hpd(post1)
plt.show()
'''

