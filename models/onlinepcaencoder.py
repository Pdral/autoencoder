from os import listdir

from pca import pca
import numpy as np
import pandas as pd
from pyedflib import highlevel
from preprocess.preprocess import internal_preprocess
import matplotlib.pyplot as plt

print("Início da execução online\n")

# Nessa simulação, usaremos a leitura dos arquivos como a recepção do sinal

path = "../chb01"
files = listdir(path)
empty_buffer = True
trn = True
tol = 0.99
Ntrn = 1
fs = 256
N = 2*fs
e2tst = -1

plt.ion()
fig, ax = plt.subplots()
x_data = range(N)
y_data = [None] * N
line, = ax.plot(x_data, y_data, label='Sinal em tempo real', color='green')
line2, = ax.plot(x_data, y_data, color='green')
line.set_xdata(x_data)
line2.set_xdata(x_data)
ax.set_xlim(0, N)
ax.set_title("Canal 1")
ax.set_xlabel("Tempo")
ax.set_ylabel("Sinal")
plt.legend()

for file in files:
  path_to_file = path + "/" + file + "/" + file + ".edf"
  signals, signal_headers, header = highlevel.read_edf(path_to_file)
  for i in range(len(signals[0])):
    new_signals = signals[:, i]
    if (empty_buffer):
      buffer = np.array(new_signals).reshape((-1, 1))
      empty_buffer = False
    else:
      buffer = np.column_stack((buffer, new_signals))
      print(buffer.shape)
    if(trn and buffer.shape[1] == Ntrn*N*1800):
      Xtrn = internal_preprocess(buffer)
      V, VEi = pca(Xtrn)
      VE = np.cumsum(VEi)
      q = np.where(VE >= tol)[0][0] + 1
      Vq = V.iloc[:, :q]
      Q = Vq.T
      Ztrn = np.dot(Q, Xtrn)
      Xrectrn = np.dot(Q.T, Ztrn)
      Etrn = Xtrn - Xrectrn
      e2trn = []
      for i in range(Xtrn.shape[1]):
        e2trn.append(np.dot(Etrn.iloc[:, i].T, Etrn.iloc[:, i]))
      L = np.percentile(e2trn, 95)
      empty_buffer = True
      trn = False
      print("Fim do treino")
    if(not trn and buffer.shape[1] == N):
      print("Entrei no teste")
      Xtst = internal_preprocess(buffer)
      Ztst = np.dot(Q, Xtst)
      Xrectst = np.dot(Q.T, Ztst)
      Etst = pd.DataFrame(Xtst - Xrectst)
      old_e2tst = e2tst
      e2tst = np.dot(Etst.T, Etst)
      signal = buffer[0]
      for i in range(N):
        if i == 0:
          if e2tst > 0:
            line.set_color(color='red')
          else:
            line.set_color(color='green')
          if old_e2tst > 0:
            line2.set_color(color='red')
          else:
            line2.set_color(color='green')
          z_data = y_data
          y_data = [None] * N

        y_data[i] = signal[i]
        z_data[i] = None

        line.set_ydata(y_data)
        line2.set_ydata(z_data)

        ax.relim()
        ax.autoscale_view()
        plt.pause(10 / N)

      plt.ioff()
      plt.show()