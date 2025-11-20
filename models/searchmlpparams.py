import pickle
from os import listdir
import numpy as np
import pandas as pd
import math
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

data_path = '../data'
patients  = listdir(data_path)
for patient in patients:

    print(f'\nPaciente {patient}\n')

    print("Lendo arquivos")

    path = "../data/" + patient
    files = listdir(path)
    signals = []
    labels = []
    tol = 0.99
    names = ['Acurácia', 'Num_VP', 'VP_rate', 'Num_VN', 'VN_rate', 'Num_FP', 'FP_rate',
             'Num_FN', 'FN_rate', 'ACC2', 'Sensibilidade', 'Especificidade', 'MG', 'Precisão', 'F1']
    metrics_names = ['Acurácia', 'Sensibilidade', 'Especificidade']
    mean_metrics = []
    std_metrics = []

    # Lendo arquivos de atributos e labels
    for file in files:
      path_to_file = path + "/" + file + "/" + file
      with open(path_to_file + "-cov.txt", 'rb') as fp:
        signals.append(pickle.load(fp))
      with open(path_to_file + "-labels.txt", 'rb') as fp:
        labels.append(pickle.load(fp))

    # Os Nfiles primeiros arquivos serão utilizados para treino
    Nfiles = 4
    Nrodadas = 50

    # Separando dados de treino e teste
    test_signals = signals.copy()
    Xtrn = test_signals.pop(0)
    test_labels = labels.copy()
    test_labels.pop(0)
    for i in range(Nfiles-1):
      Xtrn = pd.DataFrame(np.column_stack((Xtrn, test_signals.pop(0))))
      test_labels.pop(0)

    Xtrn = Xtrn.T
    N = range(5,106,10)
    for n in N:
        all_metrics = pd.DataFrame(columns=names)
        print(f"\n{n} neurônios")
        for i in range(Nrodadas):
            # Treino
            print(f"Rodada {i+1}/{Nrodadas}")
            metrics = []
            mlp = MLPRegressor(hidden_layer_sizes=(138, n, 138), activation='logistic', solver='adam',
                               max_iter=1000, early_stopping=True, tol=1e-5,
                               learning_rate='adaptive', batch_size=64)
            mlp.fit(Xtrn, Xtrn)
            Xrectrn = mlp.predict(Xtrn)
            Etrn = (Xtrn - Xrectrn).T
            e2trn = []
            for i in range(Xtrn.shape[0]):
                e2trn.append(np.dot(Etrn.iloc[:, i].T, Etrn.iloc[:, i]))
            L = np.percentile(e2trn, 95)

            # Teste
            Xtst = np.column_stack(test_signals).T
            Xrectst = mlp.predict(Xtst)
            Etst = pd.DataFrame((Xtst - Xrectst).T)
            e2tst = []
            for i in range(Xtst.shape[0]):
                e2tst.append(np.dot(Etst.iloc[:, i].T, Etst.iloc[:, i]))

            # Métricas
            Ipred_pos = list(np.where(e2tst > L)[0])
            Ypred_all = -np.ones(Xtst.shape[0])
            Ypred_all[Ipred_pos] = 1

            Ytst = np.array([
                x
                for xs in test_labels
                for x in xs
            ])

            prod_rotulos = Ytst * Ypred_all
            Num_acertos_total = len(np.where(prod_rotulos > 0)[0])
            ACC1 = 100 * Num_acertos_total / len(Ytst)
            metrics.append(ACC1)

            Num_VP = len(np.where((Ytst > 0) & (Ypred_all > 0))[0])
            metrics.append(Num_VP)
            VP_rate = 100 * Num_VP / len(np.where(Ytst > 0)[0])
            metrics.append(VP_rate)

            Num_VN = len(np.where((Ytst < 0) & (Ypred_all < 0))[0])
            metrics.append(Num_VN)
            VN_rate = 100 * Num_VN / len(np.where(Ytst < 0)[0])
            metrics.append(VN_rate)

            Num_FP = len(np.where((Ytst < 0) & (Ypred_all > 0))[0])
            metrics.append(Num_FP)
            FP_rate = 100 * Num_FP / len(np.where(Ytst < 0)[0])
            metrics.append(FP_rate)

            Num_FN = len(np.where((Ytst > 0) & (Ypred_all < 0))[0])
            metrics.append(Num_FN)
            FN_rate = 100 * Num_FN / len(np.where(Ytst > 0)[0])
            metrics.append(FN_rate)

            ACC2 = 100 * (Num_VP + Num_VN) / (Num_VP + Num_VN + Num_FP + Num_FN)
            metrics.append(ACC2)

            Sensibilidade = 100 * Num_VP / (Num_VP + Num_FN)
            metrics.append(Sensibilidade)

            Especificidade = 100 * Num_VN / (Num_VN + Num_FP)
            metrics.append(Especificidade)

            MG = math.sqrt(Sensibilidade * Especificidade)
            metrics.append(MG)

            Precisao = 100 * Num_VP / (Num_VP + Num_FP)
            metrics.append(Precisao)

            F1 = 2 * Precisao * Sensibilidade / (Precisao + Sensibilidade)
            metrics.append(F1)

            all_metrics.loc[len(all_metrics)] = metrics

        # Dados das métricas obtidas
        final_metrics = pd.DataFrame([all_metrics.mean(), all_metrics.std(), all_metrics.max(),
                                      all_metrics.min(), all_metrics.median()], columns=names)
        final_metrics.index = ['Média', 'Desvio Padrão', 'Máximo', 'Mínimo', 'Mediana']
        # final_metrics.index = [str(int(n)) + " neurônios" for n in N ]
        final_metrics.to_csv('../metrics/search/triple-' + patient + '-' + str(n) + 'neuronios.csv')
        mean_metrics.append(final_metrics.loc['Média', metrics_names])
        std_metrics.append(final_metrics.loc['Desvio Padrão', metrics_names])
    print("\nPlotando")
    plot_mean_metrics = pd.DataFrame(mean_metrics, columns=metrics_names)
    plot_std_metrics = pd.DataFrame(std_metrics, columns=metrics_names)
    for metric_name in metrics_names:
        plt.xlabel("Neurônios")
        plt.ylabel(metric_name)
        plt.title(metric_name + " no paciente " + patient.upper())
        plt.errorbar(N, plot_mean_metrics.loc[:, metric_name], yerr=plot_std_metrics.loc[:, metric_name], capsize=5.0)
        plt.savefig('../graphs/triple-' + metric_name + '-' + patient + '.png')
        plt.clf()