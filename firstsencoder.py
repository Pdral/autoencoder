import pickle
from os import listdir
from models.pca import pca
import numpy as np
import pandas as pd
import math
from models.split_data import split_data
from models.online_pca_encoder import online_pca_encoder
from models.online_mlp_encoder import online_mlp_encoder
from metrics.metrics_calculator import calculate_metrics

absolute_path = "data"
patients = listdir(absolute_path)
# patients = ['chb18']

for patient in patients:
    print(f'Paciente {patient}\n')
    path = "data/" + patient
    extensions = [
        "-cov.txt",
        "-time_features.txt",
        "-freq_features.txt"
    ]
    metrics = []

    # Os Nfiles primeiros arquivos serão utilizados para treino
    Nfiles = 4

    Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles, True)
    attr_size = Xtrn.shape[0]

    # Definição de modelos
    models = [
        online_pca_encoder(tol = 0.99),
        online_mlp_encoder(hidden_layer_sizes = (120,)),
        online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2)))
    ]

    Yens = 0

    for model in models:

        print(f'Modelo {model.name}')

        # Treino
        print("Treino")
        model.train(Xtrn)

        # Teste
        print("Teste")
        Ypred = model.test(Xtst)
        Yens += Ypred

        # Métricas
        print("Calculando métricas\n")
        metrics.append(calculate_metrics(Ytst, Ypred))

    # Avaliação do comitê
    print(f'Modelo comitê')
    print("Calculando métricas\n")
    metrics.append(calculate_metrics(Ytst, Yens))

    # Dados das métricas obtidas
    final_metrics = pd.DataFrame(metrics)
    final_metrics.index = [model.name for model in models] + ['ensemble']
    final_metrics.to_csv('metrics/reduced/cov-time-freq/metrics-' + patient + '-online.csv')
    # with open('metrics.txt', 'wb') as fp:
    #   pickle.dump(final_metrics, fp)