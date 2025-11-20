import pandas as pd
import numpy as np
import math
from models.split_data import split_data
from models.online_mlp_encoder import online_mlp_encoder
from metrics.metrics_calculator import calculate_metrics
import matplotlib.pyplot as plt

patients = [
    'chb03',
    'chb05',
    'chb07',
    'chb14',
    'chb18'
]

for patient in patients:
    # Definição dos dados
    path = "data/" + patient
    extensions = [
        # "-cov.txt",
        # "-time_features.txt",
        "-freq_features.txt"
    ]
    metrics = []
    metrics_names = ['Precision', 'Sensitivity']

    # Os Nfiles primeiros arquivos serão utilizados para treino
    Nfiles = 4

    Xtrn, (Xtst, Ytst) = split_data(path, extensions, Nfiles, reduce=True)
    attr_size = Xtrn.shape[0]
    percs = np.arange(95, 100, 0.1)

    for perc in percs:
        print(f'Percentil do erro = {perc}\n')
        
        # Definição de modelos
        model = online_mlp_encoder(hidden_layer_sizes = (math.ceil(attr_size/2), math.ceil(attr_size/4), math.ceil(attr_size/2)))

        # Treino
        print("Treino")
        model.train(Xtrn, perc=perc)

        # Teste
        print("Teste")
        Ypred = model.test(Xtst)

        # Métricas
        print("Calculando métricas\n")
        metrics.append(calculate_metrics(Ytst, Ypred))

    # Dados das métricas obtidas
    final_metrics = pd.DataFrame(metrics)
    final_metrics.index = [perc for perc in percs]
    final_metrics.to_csv('graphs/search/perc/'+ patient +'/perc_study.csv')

    print("\nPlotando")
    for metric_name in metrics_names:
        plt.xlabel("Percentil")
        plt.ylabel(metric_name)
        plt.title(metric_name + " x Percentil")
        plt.plot(percs, final_metrics[metric_name])
        plt.savefig('graphs/search/perc/' + patient + '/perc_' + metric_name + '_study.png')
        plt.clf()