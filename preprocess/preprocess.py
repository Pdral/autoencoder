import pandas as pd
from pyedflib import highlevel
import numpy as np
import math, time
from freq_feature import extract_features
from joblib import Parallel, delayed
from functools import partial

def preprocess(path):
    signals, signal_headers, header = highlevel.read_edf(path)
    return internal_preprocess(signals)

def internal_preprocess(signals):

    start = time.time()

    signals = signals[list(np.where(np.std(signals, axis=1))[0])]

    n = 512
    N = len(signals[0])
    E = math.ceil(N / n)

    partial_generate_matrix = partial(generate_matrix, signals=signals, n=n)
    features_list = Parallel(n_jobs=-1)(delayed(partial_generate_matrix)(k) for k in range(E))
    
    matrix = np.column_stack(features_list)        

    me = np.mean(matrix, axis=1).reshape(-1, 1)
    se = np.std(matrix, axis=1).reshape(-1, 1)

    end = time.time()
    print(f'Duração total para processar {E} épocas: {(end-start):.2f}s')

    return pd.DataFrame(((matrix - me) / se)).fillna(0)

def generate_matrix(k, signals, n):
    try:
        signal = signals[:, n * k:n * (k + 1)]
    except:
        signal = signals[:, n * k:]

    features = extract_features(signal)
    
    return features