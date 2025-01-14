import pickle
from os import listdir
from pca import pca
import numpy as np
import pandas as pd
import math

print("Lendo arquivos\n")

path = "../chb01"
files = listdir(path)
signals = []
labels = []
tol = 0.99
names = ['q', 'L', 'ACC1', 'Num_VP', 'VP_rate', 'Num_VN', 'VN_rate', 'Num_FP', 'FP_rate',
         'Num_FN', 'FN_rate', 'ACC2', 'Sensibilidade', 'Especificidade', 'MG', 'Precisão', 'F1']
all_metrics = pd.DataFrame(columns=names)

# Lendo arquivos de atributos e labels
for file in files:
  path_to_file = path + "/" + file + "/" + file
  with open(path_to_file + "-cov.txt", 'rb') as fp:
    signals.append(pickle.load(fp))
  with open(path_to_file + "-labels.txt", 'rb') as fp:
    labels.append(pickle.load(fp))

metrics = []
# Os Nfiles primeiros arquivos serão utilizados para treino
Nfiles = 4

# Separando dados de treino e teste
test_signals = signals.copy()
Xtrn = test_signals.pop(0)
test_labels = labels.copy()
test_labels.pop(0)
for i in range(Nfiles-1):
  Xtrn = pd.DataFrame(np.column_stack((Xtrn, test_signals.pop(0))))
  test_labels.pop(0)

# Treino
print("Treino\n")
V, VEi = pca(Xtrn)
VE = np.cumsum(VEi)
q = np.where(VE >= tol)[0][0] + 1
metrics.append(q)
Vq = V.iloc[:, :q]
Q = Vq.T
Ztrn = np.dot(Q, Xtrn)
Xrectrn = np.dot(Q.T, Ztrn)
Etrn = Xtrn - Xrectrn
e2trn = []
for i in range(Xtrn.shape[1]):
  e2trn.append(np.dot(Etrn.iloc[:, i].T, Etrn.iloc[:, i]))
L = np.percentile(e2trn, 95)
metrics.append(L)

# Teste
print("Teste\n")
Xtst = np.column_stack(test_signals)
Ztst = np.dot(Q, Xtst)
Xrectst = np.dot(Q.T, Ztst)
Etst = pd.DataFrame(Xtst - Xrectst)
e2tst = []
for i in range(Xtst.shape[1]):
  e2tst.append(np.dot(Etst.iloc[:, i].T, Etst.iloc[:, i]))

# Métricas
Ipred_pos = list(np.where(e2tst > L)[0])
Ypred_all = -np.ones(Xtst.shape[1])
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
final_metrics.to_csv('metrics-online' + str(Nfiles) +  '.csv')
# with open('metrics.txt', 'wb') as fp:
#   pickle.dump(final_metrics, fp)