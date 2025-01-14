import scipy.io
import pandas as pd
import numpy as np
from math import floor

def preprocessMat():
    X = scipy.io.loadmat("deteccao de anomalias - PCA/chb03/hipotese4/X.mat")['X']
    Y = scipy.io.loadmat("deteccao de anomalias - PCA/chb03/hipotese4/Y.mat")['Y'][0]
    X = pd.DataFrame(X)
    Ipos = np.where(Y > 0)[0]
    Npos = len(Ipos)
    Xpos = X.iloc[:, Ipos]
    Ypos = list(Y[Ipos])
    me = np.array(np.mean(Xpos, axis=1)).reshape(-1, 1)
    se = np.array(np.std(Xpos, axis=1)).reshape(-1, 1)
    Xpos = pd.DataFrame((Xpos-me)/se)

    Ineg = np.where(Y < 0)[0]
    Nneg = len(Ineg)
    Xneg = X.iloc[:, Ineg]
    Yneg = list(Y[Ineg])
    me = np.array(np.mean(Xneg, axis=1)).reshape(-1, 1)
    se = np.array(np.std(Xneg, axis=1, ddof=1)).reshape(-1, 1)
    Xneg = pd.DataFrame((Xneg-me)/se)

    Nneg_trn = floor(0.8*Nneg)
    Xneg_trn = Xneg.iloc[:, :Nneg_trn]
    Xneg_tst = Xneg.iloc[:, Nneg_trn:]
    Yneg_tst = Yneg[Nneg_trn:]
    Nneg_tst = Nneg-Nneg_trn

    Xtst=np.column_stack((Xpos, Xneg_tst))
    Ytst=np.array(Ypos + Yneg_tst)
    Ntst=Xtst.shape[1]

    return Xneg_trn, Xtst, Ytst