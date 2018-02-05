import pandas as pd
import numpy as np 
from numpy import genfromtxt
#path = "hmc3/hmc_"
path = "../build/mhmc_mnist"

prob_mean = genfromtxt(path+"/predict_proba_mean.csv", delimiter=",")
prob_std = genfromtxt(path+"/predict_proba_std.csv", delimiter=",")
prob_max = genfromtxt(path+"/predict_proba_max.csv", delimiter=",")
gt = genfromtxt(path+"/Y_test.csv", delimiter=",")
classes = genfromtxt(path+"/classes.csv", delimiter=",")

mean_header =["mean_"+s for s in map(str,classes)]
std_header =["std_"+s for s in map(str,classes)]
max_gt_header = ["max_prob", "GT"]
header = np.concatenate((mean_header, max_gt_header, std_header), axis=0)
result = np.concatenate((prob_mean, np.reshape(prob_max,(-1,1)), np.reshape(gt,(-1,1)),prob_std),axis=1)
np.savetxt(path+"/mnist_analysis.csv", result, fmt="%1.3f", header = ", ".join(str(x) for x in header),delimiter = ",")



