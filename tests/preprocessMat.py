import pandas as pd
from pyedflib import highlevel
import numpy as np
import math
from os import listdir
import pickle

def preprocess(path):
    signals, signal_headers, header = highlevel.read_edf(path)
    return internal_preprocess(signals)

def internal_preprocess(signals):
    signals = signals[list(np.where(np.std(signals, axis=1))[0])]
    n = 512
    N = len(signals[0])
    E = math.ceil(N / n)

    print(len(signals))

    for k in range(E):
        try:
            cov = np.cov(signals[:, n * k:n * (k + 1)])
        except:
            cov = np.cov(signals[:, n * k:])
        vet = []
        p = len(cov)
        for i in range(p):
            for j in range(p):
                if i == j:
                    vet.append(cov[i, j])
                elif i > j:
                    vet.append(math.sqrt(2) * cov[i, j])
        v = np.array(vet).reshape(-1, 1)
        if k == 0:
            matrix = v
            a = len(v)
        else:
            matrix = np.column_stack((matrix, v))
    me = np.mean(matrix, axis=1).reshape(-1, 1)
    se = np.std(matrix, axis=1).reshape(-1, 1)
    return pd.DataFrame((matrix - me)/ se).dropna(axis = 0, how = 'any')

if __name__ == "__main__":
    path = "../data/chb14"

    name = "chb14_01"
    dir = path + "/" + name + "/"
    signal = preprocess(dir + name + ".edf")
    # with open(dir + name + "-cov.txt", 'wb') as fp:
    #     pickle.dump(signal, fp)