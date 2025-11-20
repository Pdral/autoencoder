from os import listdir
import pickle
import pandas as pd
import numpy as np
from models.pca import pca

def split_data(path, extensions, Nfiles, reduce=False):
    print("Lendo arquivos\n")

    files = listdir(path)
    signals = []
    labels = []

    # Lendo arquivos de atributos e labels
    for file in files:
        path_to_file = path + "/" + file + "/" + file
        file_signals = []
        for file_extension in extensions:
            with open(path_to_file + file_extension, 'rb') as fp:
                file_signals.append(pickle.load(fp))
        signals.append(pd.concat(file_signals))
        with open(path_to_file + "-labels.txt", 'rb') as fp:
            labels.append(pickle.load(fp))

    # Separando dados de treino e teste
    min_length = min(len(arr) for arr in signals)
    signals = [arr[:min_length].reset_index(drop=True) for arr in signals]
    # signals = [signal for signal in signals]

    if reduce:
        Xtrn, test_signals = reduce_func(signals, Nfiles)
    else:
        Xtrn = pd.concat(signals[:Nfiles], axis=1)
        test_signals = pd.concat(signals[Nfiles:], axis=1)
    
    test_labels = labels[Nfiles:]

    Ytst = np.array([
        x
        for xs in test_labels
        for x in xs
    ])

    return Xtrn, (test_signals, Ytst)

def reduce_func(signals_list, Nfiles, tol=0.99):
    n = 0
    for i in range(Nfiles): n += signals_list[i].shape[1]
    signals = pd.concat(signals_list, axis=1)
    antes = signals.shape
    V, VEi = pca(signals)
    VE = np.cumsum(VEi)
    q = np.where(VE >= tol)[0][0] + 1
    Vq = V.iloc[:, :q]
    Q = Vq.T
    Z = np.dot(Q, signals)
    X = pd.DataFrame(Z)
    depois = X.shape
    print(f'Antes: {antes} - Depois: {depois}')
    return X.iloc[:, :n], X.iloc[:, n:]