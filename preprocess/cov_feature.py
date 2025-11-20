import numpy as np
import math, time

def extract_features(signal):
    start = time.time()
    cov = np.cov(signal)
    vet = []
    p = len(cov)
    for i in range(p):
        for j in range(p):
            if i == j:
                vet.append(cov[i, j])
            elif i > j:
                vet.append(math.sqrt(2) * cov[i, j])
    end = time.time()
    print(f'Processamento da época ocorreu em {(end-start):.2f}s')
    return np.array(vet).reshape(-1, 1)