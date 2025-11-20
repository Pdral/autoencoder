import pickle
from os import listdir
import numpy as np
import pandas as pd
import math
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt



N = range(5,106,10)

mean_metrics = []
std_metrics = []
metrics_names = ['Acurácia', 'Sensibilidade', 'Especificidade']
files = listdir("../metrics/search")
for file in files:
    if file.startswith("chb18"):
        final_metrics = pd.read_csv("../metrics/search/" + file)
        final_metrics.index = final_metrics.iloc[:, 0]
        final_metrics.drop(final_metrics.columns[[0]], axis=1, inplace=True)
        mean_metrics.append(final_metrics.loc['Média', metrics_names])
        std_metrics.append(final_metrics.loc['Desvio Padrão', metrics_names])

print("\nPlotando")
plot_mean_metrics = pd.DataFrame(mean_metrics, columns=metrics_names).iloc[[6,1,2,3,4,5,7,8,9,10,0], :]
plot_std_metrics = pd.DataFrame(std_metrics, columns=metrics_names).iloc[[6,1,2,3,4,5,7,8,9,10,0], :]
for metric_name in metrics_names:
    plt.xlabel("Neurônios")
    plt.ylabel(metric_name)
    plt.title(metric_name + " no paciente CHB18")
    plt.errorbar(N, plot_mean_metrics.loc[:, metric_name], yerr=plot_std_metrics.loc[:, metric_name], capsize=5.0)
    plt.savefig('../graphs/' + metric_name + '-chb182.png')
    plt.clf()