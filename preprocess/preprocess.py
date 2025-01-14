import pandas as pd
from pyedflib import highlevel
import numpy as np
import math

def preprocess(path):
    signals, signal_headers, header = highlevel.read_edf(path)
    return internal_preprocess(signals)

def internal_preprocess(signals):
    n = 512
    N = len(signals[0])
    E = math.ceil(N / n)

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
        else:
            matrix = np.column_stack((matrix, v))

    me = np.mean(matrix, axis=1).reshape(-1, 1)
    se = np.std(matrix, axis=1).reshape(-1, 1)
    return pd.DataFrame((matrix - me) / se)