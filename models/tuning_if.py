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

param_grid_iso = {
    'n_estimators': [10, 70, 100, 200, 300], 
    
    'max_samples': [128, 256, 512], 
    
    'contamination': [0.01, 0.05, 0.1, 0.15, 0.3, 0.4], 
    
    'max_features': [1.0, 0.8, 0.5]
}

extensions = ["-cov.txt"]

path = "data/chb01"
extensions_names = ['cov']

# Os Nfiles primeiros arquivos serão utilizados para treino
Nfiles = 4

Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles)
attr_size = Xtrn.shape[0]

n_rounds = len(param_grid_iso['n_estimators'])*len(param_grid_iso['max_samples'])*len(param_grid_iso['contamination'])*len(param_grid_iso['max_features'])
i = 0

for n_estimators in param_grid_iso['n_estimators']:
    metrics = []
    params = []
    for max_samples in param_grid_iso['max_samples']:
        for contamination in param_grid_iso['contamination']:
            for max_features in param_grid_iso['max_features']:
                i += 1
                print(f'Rodada {i}/{n_rounds}')
                model = online_if_detector(n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, max_features=max_features)

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
                params.append(str(max_samples) + '-' + str(contamination) + '-' + str(max_features))
                print(f"Sensibilidade: {calculated_metrics['Sensitivity']}\n")

    # Dados das métricas obtidas
    final_metrics = pd.DataFrame(metrics)
    final_metrics.index = [params]
    final_metrics.to_csv(f'metrics/search/tuning/if/estimadores_{str(n_estimators)}.csv')