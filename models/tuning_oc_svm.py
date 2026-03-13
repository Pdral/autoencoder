import pickle
from os import listdir
from models.pca import pca
import re
import numpy as np
import pandas as pd
import math
from itertools import combinations
from models.split_data import split_data
from models.online_pca_encoder import online_pca_encoder
from models.online_mlp_encoder import online_mlp_encoder
from models.online_oc_svm_detector import online_oc_svm_detector
from models.online_if_detector import online_if_detector
from metrics.metrics_calculator import calculate_metrics

param_grid_svm = {
    'kernel': ['sigmoid', 'poly'], 
    
    'nu': [0.01, 0.05, 0.1, 0.15, 0.3, 0.4], 
    
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 0.5, 1.0]
}

extensions = ["-cov.txt"]

path = "data/chb01"
extensions_names = ['cov']

# Os Nfiles primeiros arquivos serão utilizados para treino
Nfiles = 4

Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles)
attr_size = Xtrn.shape[0]

n_rounds = len(param_grid_svm['gamma'])*len(param_grid_svm['nu'])*len(param_grid_svm['kernel'])
i = 0

for kernel in param_grid_svm['kernel']:
    metrics = []
    params = []
    for gamma in param_grid_svm['gamma']:
        for nu in param_grid_svm['nu']:
            i += 1
            print(f'Rodada {i}/{n_rounds}')
            model = online_oc_svm_detector(kernel=kernel, gamma=gamma, nu=nu)

            # Treino
            print("Treino")
            model.train(Xtrn)

            # Teste
            print("Teste")
            Ypred = model.test(Xtst)

            # Métricas
            print("Calculando métricas")
            calculated_metrics = calculate_metrics(Ytst, Ypred)
            metrics.append(calculated_metrics)
            params.append(str(kernel) + '-' + str(gamma) + '-' + str(nu))
            print(f"Sensibilidade: {calculated_metrics['Sensitivity']}\n")

    # Dados das métricas obtidas
    final_metrics = pd.DataFrame(metrics)
    final_metrics.index = [params]
    final_metrics.to_csv(f'metrics/search/tuning/oc_svm/{kernel}.csv')