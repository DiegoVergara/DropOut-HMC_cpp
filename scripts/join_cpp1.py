import pandas as pd
#path = "hmc3/hmc_"
path = "hmc1/hmc_"

mean_0 = pd.read_csv(path+"mean_0.csv", header = None, names =['mean_0'])
mean_1 = pd.read_csv(path+"mean_1.csv", header = None, names =['mean_1'])
mean_2 = pd.read_csv(path+"mean_2.csv", header = None, names =['mean_2'])
mean_3 = pd.read_csv(path+"mean_3.csv", header = None, names =['mean_3'])
mean_4 = pd.read_csv(path+"mean_4.csv", header = None, names =['mean_4'])
mean_5 = pd.read_csv(path+"mean_5.csv", header = None, names =['mean_5'])
mean_6 = pd.read_csv(path+"mean_6.csv", header = None, names =['mean_6'])
mean_7 = pd.read_csv(path+"mean_7.csv", header = None, names =['mean_7'])
mean_8 = pd.read_csv(path+"mean_8.csv", header = None, names =['mean_8'])
mean_9 = pd.read_csv(path+"mean_9.csv", header = None, names =['mean_9'])

v0 = pd.read_csv(path+"mean_0.csv", header = None, names =['0'])
v1 = pd.read_csv(path+"mean_1.csv", header = None, names =['1'])
v2 = pd.read_csv(path+"mean_2.csv", header = None, names =['2'])
v3 = pd.read_csv(path+"mean_3.csv", header = None, names =['3'])
v4 = pd.read_csv(path+"mean_4.csv", header = None, names =['4'])
v5 = pd.read_csv(path+"mean_5.csv", header = None, names =['5'])
v6 = pd.read_csv(path+"mean_6.csv", header = None, names =['6'])
v7 = pd.read_csv(path+"mean_7.csv", header = None, names =['7'])
v8 = pd.read_csv(path+"mean_8.csv", header = None, names =['8'])
v9 = pd.read_csv(path+"mean_9.csv", header = None, names =['9'])


std_0 = pd.read_csv(path+"std_0.csv", header = None, names =['std_0'])
std_1 = pd.read_csv(path+"std_1.csv", header = None, names =['std_1'])
std_2 = pd.read_csv(path+"std_2.csv", header = None, names =['std_2'])
std_3 = pd.read_csv(path+"std_3.csv", header = None, names =['std_3'])
std_4 = pd.read_csv(path+"std_4.csv", header = None, names =['std_4'])
std_5 = pd.read_csv(path+"std_5.csv", header = None, names =['std_5'])
std_6 = pd.read_csv(path+"std_6.csv", header = None, names =['std_6'])
std_7 = pd.read_csv(path+"std_7.csv", header = None, names =['std_7'])
std_8 = pd.read_csv(path+"std_8.csv", header = None, names =['std_8'])
std_9 = pd.read_csv(path+"std_9.csv", header = None, names =['std_9'])

means = pd.concat([mean_0, mean_1, mean_2, mean_3, mean_4, mean_5, mean_6, mean_7, mean_8, mean_9], axis=1, join_axes=[mean_0.index])

stds = pd.concat([std_0, std_1, std_2, std_3, std_4, std_5, std_6, std_7, std_8, std_9], axis=1, join_axes=[std_0.index])

gt = pd.read_csv("../data/MNIST/Y_test.csv", header = None, names =['gt'])

max_mean = pd.DataFrame(data = means.max(axis=1), columns=['max_mean']) 
max_std = pd.DataFrame(data = stds.max(axis=1), columns=['max_std']) 

result = pd.concat([means, max_mean, gt, stds, max_std], axis=1, join_axes=[means.index])

result.to_csv(path+"result.csv",sep=',', encoding='utf-8')

means2 = pd.concat([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9], axis=1, join_axes=[v0.index])
y_hat = means2.T.idxmax()
y_hat = y_hat.values.flatten().astype(int)

gt = gt.values.flatten().astype(int)
acc = (y_hat==gt).sum()*100/y_hat.shape[0]
print(acc)