import numpy as np
import math
import pandas as pd
from models.pca import pca
from preprocess.preprocess import preprocess
from models.realtimeplot import realtimeplot
from pyedflib import highlevel

# Xneg_trn, Xtst, Ytst = preprocessMat()

path = "../chb01/chb01_01/chb01_01.edf"

print("Treino")
Xneg_trn = preprocess(path)

V, VEi = pca(Xneg_trn)
VE = np.cumsum(VEi)
tol = 0.99
q = np.where(VE >= tol)[0][0] + 1
Vq = V.iloc[:, :q]
Q = Vq.T
Ztrn = np.dot(Q, Xneg_trn)
Xrectrn = np.dot(Q.T, Ztrn)
Etrn = Xneg_trn - Xrectrn
e2trn = []
for i in range(Xneg_trn.shape[1]):
  e2trn.append(np.dot(Etrn.iloc[:,i].T, Etrn.iloc[:,i]))
L = np.percentile(e2trn, 95)

print("Teste")

path = "chb03_01.edf"
Xtst = preprocess(path)
Ytst = -np.ones(Xtst.shape[1])
Ytst[range(int(362/2), int(414/2))] = 1
Ztst = np.dot(Q, Xtst)
Xrectst = np.dot(Q.T, Ztst)
Etst = pd.DataFrame(Xtst - Xrectst)
e2tst = []
for i in range(Xtst.shape[1]):
  e2tst.append(np.dot(Etst.iloc[:,i].T, Etst.iloc[:,i]))

Ipred_pos = list(np.where(e2tst > L)[0])
Ypred_all = -np.ones(Xtst.shape[1])
Ypred_all[Ipred_pos] = 1

prod_rotulos = Ytst*Ypred_all
Num_acertos_total = len(np.where(prod_rotulos > 0)[0])
ACC1 = 100*Num_acertos_total/len(Ytst)
print(f"ACC1: {ACC1}\n")

Num_VP = len(np.where((Ytst > 0) & (Ypred_all > 0))[0])
VP_rate = 100*Num_VP/len(np.where(Ytst > 0)[0])
print(f"Num_VP: {Num_VP}\nVP_rate: {VP_rate}\n")

Num_VN = len(np.where((Ytst < 0) & (Ypred_all < 0))[0])
VN_rate = 100*Num_VN/len(np.where(Ytst < 0)[0])
print(f"Num_VN: {Num_VN}\nVN_rate: {VN_rate}\n")

Num_FP = len(np.where((Ytst < 0) & (Ypred_all > 0))[0])
FP_rate = 100*Num_FP/len(np.where(Ytst < 0)[0])
print(f"Num_FP: {Num_FP}\nFP_rate: {FP_rate}\n")

Num_FN = len(np.where((Ytst > 0) & (Ypred_all < 0))[0])
FN_rate = 100*Num_FN/len(np.where(Ytst > 0)[0])
print(f"Num_FN: {Num_FN}\nFN_rate: {FN_rate}\n")

ACC2 = 100*(Num_VP + Num_VN)/(Num_VP + Num_VN + Num_FP + Num_FN)
print(f"ACC2: {ACC2}\n")

Sensibilidade = 100*Num_VP/(Num_VP + Num_FN)
print(f"Sensibilidade: {Sensibilidade}\n")

Especificidade = 100*Num_VN/(Num_VN + Num_FP)
print(f"Especificidade: {Especificidade}\n")

MG = math.sqrt(Sensibilidade*Especificidade)
print(f"MG: {MG}\n")

Precisao = 100*Num_VP/(Num_VP + Num_FP)
print(f"Precisao: {Precisao}\n")

F1 = 2*Precisao*Sensibilidade/(Precisao + Sensibilidade)
print(f"F1: {F1}\n")

signals, signal_headers, header = highlevel.read_edf(path)
realtimeplot(Ipred_pos, signals[0])