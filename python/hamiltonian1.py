import autograd.numpy as np
from autograd import grad
from autograd.util import quick_grad_check
from autograd.util import flatten
from autograd.util import flatten_func

def leapfrog(x, r, step_size, grad):
    x1 = x + step_size*r
    return x1

def accept(x, y, r_0, r, logp):
    E_new = energy(logp, y, r)
    E = energy(logp, x, r_0)
    A = np.min(np.array([0, E-E_new]))
    return (np.log(np.random.rand()) < A)

def energy(logp, x, r):
    return -logp(x) + 0.5*np.dot(r, r)

def initial_momentum(state):
    n = len(state)
    mu = np.zeros(n)
    cov = np.diagflat(np.ones(n))
    new=np.random.multivariate_normal(mu, cov)
    return new

class Hamiltonian1():
    def __init__(self, logp, grad, start, step_size=1, n_steps = None , path_length=1.0, **kwargs):
        self.state=start
        self.step_size = step_size
        self.tunning = True
        self.n_steps = 10
        self.path_length = path_length
        if n_steps!=None:
            self.n_steps = n_steps
            self.tunning = False
        self.grad=grad
        self.logp=logp
        self._sampled = 0.0
        self._accepted = 0.0

    def sample(self, n_iter):
        samples = np.zeros([n_iter, len(self.state)])
        for k in range(n_iter):
            #print(k)
            samples[k,:]=self.step()
        return samples

    def step(self):
        x = self.state
        step_size = self.unif(self.step_size)
        if self.tunning:
            self.n_steps = int(self.path_length/step_size)
        r0 = initial_momentum(x)
        y, r = x, r0
        r = r - step_size/2.0*self.grad(x)
        for i in range(self.n_steps):
            y = leapfrog(y, r, step_size, self.grad)
            if (i==(self.n_steps-1)):
                r = r - step_size*self.grad(x)
        r = r - step_size/2.0*self.grad(x)
        if accept(x, y, r0, r, self.logp):
            x = y
            self._accepted = self._accepted + 1.0

        self.state = x
        self._sampled = self._sampled  + 1.0
        return x

    def acceptance_rate(self):
        return (self._accepted/(self._sampled+0.0))*100

    def unif(self, step_size, elow=.9, ehigh=1.1):
        return np.random.uniform(elow, ehigh) * step_size