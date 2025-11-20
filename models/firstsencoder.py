import pickle
from os import listdir
from pca import pca
import numpy as np
import pandas as pd
import math
from split_data import split_data
import online_pca_encoder, online_mlp_encoder
from metrics.metrics_calculator import calculate_metrics

print("Lendo arquivos\n")

patient = 'chb14'
path = "data/" + patient
file_extension = "-time_features.txt"

# Os Nfiles primeiros arquivos serão utilizados para treino
Nfiles = 4

Xtrn, (Xtst, Ytst) = split_data(path, file_extension, Nfiles)
print(t)

# Definição de modelos
models = [
    online_pca_encoder(tol = 0.99),
    online_mlp_encoder(hidden_layer_sizes = ()),
    online_mlp_encoder(hidden_layer_sizes = ())
]

for model in models:
    # Treino
    print("Treino\n")
    q, L, Q = model.train(Xtrn)

    # Teste
    print("Teste\n")
    Ypred = model.test(Q, Xtst)

    # Métricas
    print("Calculando métricas\n")
    metrics = calculate_metrics(Ytst, Ypred)

    # Dados das métricas obtidas
    final_metrics = pd.DataFrame(metrics)
    final_metrics.index = ['Medição']
    final_metrics.to_csv('metrics/time/metrics-' + patient + '-online-' + model.name + '.csv')
    # with open('metrics.txt', 'wb') as fp:
    #   pickle.dump(final_metrics, fp)