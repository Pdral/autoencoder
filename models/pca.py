import numpy as np
import pandas as pd

def pca(data):
    autoval, autovet = np.linalg.eigh(np.cov(data))
    autovet = autovet.T
    autoval = autoval / np.sum(autoval)
    order = np.argsort(autoval)[::-1]
    V = pd.DataFrame(autovet[order].T)
    VEi = autoval[order]
    return V, VEi