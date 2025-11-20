from models.pca import pca
import numpy as np
import pandas as pd

class online_pca_encoder():

    def __init__(self, tol):
        self.Q = None
        self.tol = tol
        self.L = None
        self.name = 'pca'

    def train(self, Xtrn, perc=95):
        V, VEi = pca(Xtrn)
        VE = np.cumsum(VEi)
        q = np.where(VE >= self.tol)[0][0] + 1
        Vq = V.iloc[:, :q]
        self.Q = Vq.T
        Ztrn = np.dot(self.Q, Xtrn)
        Xrectrn = np.dot(self.Q.T, Ztrn)
        Etrn = Xtrn - Xrectrn
        e2trn = []
        for i in range(Xtrn.shape[1]):
            e2trn.append(np.dot(Etrn.iloc[:, i].T, Etrn.iloc[:, i]))
        self.L = np.percentile(e2trn, perc)

    def test(self, Xtst):
        Ztst = np.dot(self.Q, Xtst)
        Xrectst = np.dot(self.Q.T, Ztst)
        Etst = pd.DataFrame(Xtst - Xrectst)
        e2tst = []
        for i in range(Xtst.shape[1]):
            e2tst.append(np.dot(Etst.iloc[:, i].T, Etst.iloc[:, i]))
        Ipred_pos = list(np.where(e2tst > self.L)[0])
        Ypred_all = -np.ones(Xtst.shape[1])
        Ypred_all[Ipred_pos] = 1
        return Ypred_all