from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression #for comparison
from sklearn.cross_validation import train_test_split
import numpy as np
from autograd.scipy.misc import logsumexp

class LogisticClassifier(object):
    """
    Multiclass logistic regression with regularization. Trained with gradient descent + momentum (if desired). 
    """
    
    def __init__(self, basis=None):
        """
        Instantiate a logistic regression model. Specify a custom basis function here,
        it should accept an array and output a (preferably higher dimensional) array.
        The default is the identity function plus a bias term.
        """
        self.W = []
        self.A = None #the mixing matrix for basis mapping.
        self.basis=basis
        if basis == 'poly':
            self.basisfunc = self.poly_basis
        elif basis == 'rbf':
            self.basisfunc = self.rbf_basis
        elif basis == 'sigmoid':
            self.basisfunc = self.sigmoid_basis
        elif basis == 'rectifier':
            self.basisfunc = self.rectifier_basis
        else:
            self.basisfunc = self.identity
            
    
    def identity(self, x):
        #identity basis function + a bias
        return np.hstack((x,1))
    
    def poly_basis(self, x):
        #polynomial basis
        degree = 2
        #first mix the components of x in a higher dimension
        xn = dot(self.A,x)
        return self.identity(hstack(tuple(sum(xn**i for i in range(degree)))))
        
    def rbf_basis(self, x):
        #in this case, use the mixing matrix as centroids.
        return self.identity(hstack(tuple(exp(-norm(x-mu)) for mu in self.A)))
    
    def sigmoid_basis(self, x):
        #just like a neural network layer.
        xn = dot(self.A, x)
        return self.identity((1+exp(-xn))**-1)
    
    def rectifier_basis(self, x):
        #used in the latest neural nets
        xn = dot(self.A, x)
        return self.identity(maximum(xn, 0))
    
    def basismap(self, X):
        #if X is an observation matrix (examples by dimensions),
        #return each row mapped to a higher dimsional space
        new_dimensions = self.basisfunc(X[0,:]).shape[0]
        Xn = np.zeros((X.shape[0], new_dimensions))
        for i,xi in enumerate(X):
            Xn[i,:] = self.basisfunc(xi)
        return Xn
    
    def fit(self, X, Y, itrs=100, learn_rate=0.1, reg=0.1,
            momentum=0.5, report_cost=False, proj_layer_size=10):
        """
        Fit the model. 
        X - observation matrix (observations by dimensions)
        Y - one-hot target matrix (examples by classes)
        itrs - number of iterations to run
        learn_rate - size of step to use for gradient descent
        reg - regularization penalty (lambda above)
        momentum - weight of the previous gradient in the update step
        report_cost - if true, return the loss function at each step (expensive).
        proj_layer_size - number of dimensions in the projection (mixing) layer. Higher -> more variance
        """
        
        #first map to a new basis
        if self.basis != 'rbf':
            self.A = np.random.uniform(-1, 1, (proj_layer_size, X.shape[1]))
        else:
            #use the training examples as bases
            self.A = X[permutation(X.shape[0])[:proj_layer_size],:]
        Xn = self.basismap(X)
        
        #set up weights
        #self.W = np.random.uniform(-0.1, 0.1, (Y.shape[1], Xn.shape[1]))
        #self.W = np.random.uniform(-1.0, 1.0, (Y.shape[1], Xn.shape[1]))/Xn.shape[1]
        self.W = np.ones((Y.shape[1], Xn.shape[1]))
 
        #optimize
        previous_grad = np.zeros(self.W.shape) #used in momentum
        for i in range(itrs):
            if report_cost:
                #if (i%10 == 0):
                    print("Iteration: {:15}| AVG Loss : {:20}|".format(i, self.loss(X,Y,reg)))
            grad = self.grad(Xn, Y, reg) #compute gradient
            self.W = self.W - learn_rate*(grad + momentum*previous_grad) #take a step, use previous gradient as well.
            previous_grad = grad
        
    
    def softmax(self, Z):
        #returns sigmoid elementwise
        Z = np.maximum(Z, -1e3)
        Z = np.minimum(Z, 1e3)
        numerator = np.exp(Z)
        return numerator / np.sum(numerator, axis=1).reshape((-1,1))

    
    def logsoftmax(self, Z):
        z_max = np.max(Z, axis=0)
        logsumexp = np.log(np.sum(np.exp(Z - z_max), axis=0)) + z_max
        return Z - logsumexp
        #return Z - logsumexp(Z, axis=1, keepdims=True)
    
    
    def predict(self, X):
        """
        If the model has been trained, makes predictions on an observation matrix (observations by features)
        """
        Xn = self.basismap(X)
        return self.softmax(np.dot(Xn, self.W.T))
    
    def grad(self, Xn, Y, reg):
        """
        Returns the gradient of the loss function wrt the weights. 
        """
        #Xn should be the design matrix
        Yh = self.softmax(np.dot(Xn, self.W.T))
        return -np.dot(Y.T-Yh.T,Xn)/Xn.shape[0] + reg*self.W
    
    def loss(self, X, Y, reg):
        Xn = self.basismap(X)
        #assuming X is the data matrix
        #Yh = self.softmax(np.dot(Xn, self.W.T))
        Yh = self.logsoftmax(np.dot(Xn, self.W.T))
        #return -np.mean(np.mean(Y*np.log(Yh))) - reg*np.trace(np.dot(self.W,self.W.T))/self.W.shape[0]
        #return -np.mean(np.mean(Y*Yh)) - reg*np.trace(np.dot(self.W,self.W.T))/self.W.shape[0]
        return -np.sum(np.sum(Y*Yh)) - reg*np.trace(np.dot(self.W,self.W.T))

def error_rate(Yh, Y):
    return sum( np.equal(np.argmax(Yh,axis=1), np.argmax(Y,axis=1))) / float(Yh.shape[0]) 
'''
iris_data = load_iris()
X = iris_data.data
Y = iris_data.target
lb = LabelBinarizer()
Y = lb.fit_transform(Y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.25)
#ugh, sklearn doesn't take one-hot representations
Ytrain_sk = lb.inverse_transform(Ytrain)
Ytest_sk = lb.inverse_transform(Ytest)
'''
dataset = "IRIS"
import pandas as pd
lb = LabelBinarizer()
Xtrain = pd.read_csv("../data/"+dataset+"/X_train.csv", sep =",", names = None, header = None).values
Ytrain = pd.read_csv("../data/"+dataset+"/Y_train.csv", sep =",", names = None, header = None).values
Xtest = pd.read_csv("../data/"+dataset+"/X_test.csv", sep =",", names = None, header = None).values
Ytest = pd.read_csv("../data/"+dataset+"/Y_test.csv", sep =",", names = None, header = None).values

Ytrain = lb.fit_transform(Ytrain)
Ytest = lb.fit_transform(Ytest)
Ytrain_sk = lb.inverse_transform(Ytrain)
Ytest_sk = lb.inverse_transform(Ytest)

sklearn_lr = LogisticRegression()
linear_lr = LogisticClassifier()
linear_lr.fit(Xtrain,Ytrain, momentum=0.9, learn_rate=0.1,itrs=1,reg=0.0, report_cost= True)
sklearn_lr.fit(Xtrain, Ytrain_sk)
#performance
print 'Accuracy Ours:', error_rate(linear_lr.predict(Xtest), Ytest)
print 'Accuracy Sklearn:', error_rate(sklearn_lr.predict_proba(Xtest), Ytest)

