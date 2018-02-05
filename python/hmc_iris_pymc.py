import pymc3 as pm
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import theano.tensor as tt
from scipy import optimize
from pymc3 import geweke
from pymc3 import sample_ppc
from theano import shared

def naive_hpd(post):
    sns.kdeplot(post)
    HPD = np.percentile(post, [2.5, 97.5])
    plt.plot(HPD, [0, 0], label='HPD {:.2f} {:.2f}'.format(*HPD), 
      linewidth=8, color='k')
    plt.legend(fontsize=16);
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.gca().axes.get_yaxis().set_ticks([])

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

n_iter = 1000
step_size = 0.01

with pm.Model() as iris_model:
    alpha = pm.Normal('alpha', mu=0, sd=10) # prior = 1/10^2
    beta = pm.Normal('beta', mu=0, sd=10,shape=2) # prior = 1/10^2
    mu = alpha + pm.math.dot(x_0, beta)
    theta = pm.Deterministic('theta', tt.nnet.sigmoid(mu))
    bd = pm.Deterministic('bd', (-alpha -beta[0] * x_0[:,0])/beta[1])
    yl = pm.Bernoulli('yl', p=theta, observed=y_0)
    #map_estimate = pm.find_MAP
    map_estimate = pm.Normal('map_estimate', mu=0, sd=1)/x_0.shape[1]
    step = pm.HamiltonianMC(step_scale=step_size,path_length=1.0,is_cov=True)
    iris_trace = pm.sample(n_iter,step,map_estimate)
'''
varnames=['alpha','beta0', 'beta1']
chain_0=iris_trace[int(n_iter/10):]
a = pd.DataFrame(chain_0['alpha'])
b =  pd.DataFrame(list(chain_0['beta']))
chain_0 = pd.concat([a,b], axis=1)
chain_0.columns = varnames
g=sns.pairplot(chain_0)
g.savefig("pymc3.png")
plt.show()
plt.close()
#print(chain_0)


chain_0=iris_trace[1000:]
varnames=['alpha','beta']
idx = np.argsort(x_0[:,0])
bd = chain_0['bd'].mean(0)[idx]
plt.scatter(x_0[:,0], x_0[:,1], c=y_0,s=40)
plt.plot(x_0[:,0][idx], bd, color='r')
bd_hpd = pm.hpd(chain_0['bd'])[idx]
plt.fill_between(x_0[:,0][idx], bd_hpd[:,0], bd_hpd[:,1],color='r', alpha=0.5);
plt.xlabel(x_n[0], fontsize=16)
plt.ylabel(x_n[1], fontsize=16)
plt.show()
'''

pm.autocorrplot(iris_trace,varnames=['alpha'])
plt.savefig("pymc3_auto.png")
plt.show()
plt.close()

z1 = geweke(iris_trace['beta'][:,0], intervals=15)

plt.plot(z1,'o')
plt.hlines([-2, 2], 0, 15, linestyles='dotted')
plt.xlim(0, 14)
plt.ylim(-4, 4)
plt.show()
plt.close()

z2 = geweke(iris_trace['beta'][:,1], intervals=15)

plt.plot(z2,'o')
plt.hlines([-2,2], 0, 15, linestyles='dotted')
plt.xlim(0, 14)
plt.ylim(-4,4)
plt.show()





# This function calculates the correlation between two segments of a trace.
# It takes as its input a portion of the trace to be examined and a time index.
# Calculates the correlation between the segment of all but the last t elements
# and the (potentially overlapping) segment of the first elements.
# The output of this function, the correlation coefficient between the two segments,
# is a correction factor for the (potentially correlated) standard deviations.
def rhot(x, t):
    n = len(x)
    return np.corrcoef(x[0:(n-t)], x[t:n])[0,1]

# This function inplements the Geweke's diagnostic
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

z2 = Geweke(iris_trace['alpha'][int(n_iter/10):], intervals=15, length=300)
plt.plot(z2,'o')
plt.hlines([-2,2], 0, 15, linestyles='dotted')
plt.xlim(0, 14)
plt.ylim(-4,4)
plt.savefig("pymc3_geweke.png")
plt.show()
plt.close()

# Simply running PPC will use the updated values and do prediction

with iris_model:
    plants_sim = sample_ppc(iris_trace, samples=500)

sepal = np.zeros([100,2])
sepal[:,0] = np.random.uniform(4.0,8.0,size=100)
sepal[:,1] = np.random.uniform(2.0,4.0,size=100)
sepal = (sepal - sepal.mean(axis=0)) / sepal.std(axis=0)

sepalL_to_predict = sepal[:,0]
sepalW_to_predict = sepal[:,1]

sepalL = x_0[:,0]
sepalL_shared = shared(sepalL)
sepalW  = x_0[:,1]
sepalW_shared = shared(sepalW)

# Changing values here will also change values in the model
sepalL_shared.set_value(sepalL_to_predict)
sepalW_shared.set_value(sepalW_to_predict)

tr1 = iris_trace[int(n_iter/10):]
ppc = pm.sample_ppc(tr1, model=iris_model, samples=500)

plt.errorbar(x=sepalL_to_predict, y=np.asarray(ppc['yl']).mean(axis=0), yerr=np.asarray(ppc['yl']).std(axis=0), linestyle='', marker='o')
plt.plot(sepalL, y_0, 'o')
plt.xlabel('sepalL',size=15)
plt.ylabel('sepalW ',size=15)
plt.savefig("pymc3_ppc.png")
plt.show()
plt.close()


