import numpy as np

def retroactive_check(Ytst, k):
    Ypred = [0] * k
    for i in range(k, len(Ytst)):
        Ypred.append(1 if np.sum(Ytst[i-k+1:i+1])==k else -1)
    return np.array(Ypred)

def ominous_check(Ytst, k):
    Ypred = [0] * k
    for i in range(k, len(Ytst)-k):
        Ypred.append(1 if np.sum(Ytst[i-k:i+k])>=k+1 else -1)
    Ypred.extend([0] * k)
    return np.array(Ypred)