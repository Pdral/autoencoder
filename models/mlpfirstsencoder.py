import pickle
from os import listdir
import numpy as np
import pandas as pd
import math
from sklearn.neural_network import MLPRegressor
from split_data import split_data

mlp = MLPRegressor(hidden_layer_sizes=(2), activation='logistic', solver='adam',
                    max_iter=1000, random_state=42, early_stopping=True, tol=1e-5,
                    learning_rate='adaptive', batch_size=64)

path = "data/chb01"
file_extension = "-time_features.txt"

# Os Nfiles primeiros arquivos serão utilizados para treino
Nfiles = 4

Xtrn, (test_signals, test_labels) = split_data(path, file_extension, Nfiles)

names = ['ACC1', 'Num_VP', 'VP_rate', 'Num_VN', 'VN_rate', 'Num_FP', 'FP_rate',
            'Num_FN', 'FN_rate', 'ACC2', 'Sensibilidade', 'Especificidade', 'MG', 'Precisão', 'F1']
all_metrics = pd.DataFrame(columns=names)
metrics = []

# Treino
print("Treino\n")
Xtrn = Xtrn.T
mlp.fit(Xtrn, Xtrn)
Xrectrn = mlp.predict(Xtrn)
Etrn = (Xtrn - Xrectrn).T
e2trn = []
for i in range(Xtrn.shape[0]):
  e2trn.append(np.dot(Etrn.iloc[:, i].T, Etrn.iloc[:, i]))
L = np.percentile(e2trn, 95)

# Teste
print("Teste\n")
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
final_metrics = pd.DataFrame(all_metrics, columns=names)
final_metrics.index = ['Medição']
final_metrics.to_csv('../metrics/crazy-metrics-single-mlp-chb01-online' + str(Nfiles) +  '.csv')
# with open('metrics.txt', 'wb') as fp:
#   pickle.dump(final_metrics, fp)