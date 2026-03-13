from models.pca import pca
import numpy as np
import pandas as pd
from models.detector import detector
from abc import abstractmethod

class encoder(detector):

    def __init__(self, perc=98, filter=None, filter_moment=None, k=2, q=0.5):
        self.L = None
        self.perc = perc
        self.name = 'undefined' if self.name is None else self.name
        self.name = f'{self.name}-{filter}' if filter is not None else self.name
        self.k = k
        self.q = q
        self.filters = {
            'min': lambda vec: vec.rolling(window=self.k).min().fillna(0),
            'quart': lambda vec: vec.rolling(window=2*self.k + 1, center=True).quantile(self.q, interpolation='lower').fillna(0),
            None: lambda vec: vec
        }
        self.filter = self.filters.get(filter, self.filters[None])
        self.filter_moment = filter_moment

    @abstractmethod
    def tune(self, Xtrn):
        pass

    @abstractmethod
    def reconstruct(self, X):
        pass

    def calculate_loss(self, X, Xrec):
        Etst = pd.DataFrame(X - Xrec)
        e2tst = []
        for i in range(X.shape[1]):
            e2tst.append(np.dot(Etst.iloc[:, i].T, Etst.iloc[:, i]))
        return e2tst

    def set_threshold(self, e2trn):
        self.L = np.percentile(e2trn, self.perc)

    def classify(self, e2tst):
        Ipred_pos = list(np.where(e2tst > self.L)[0])
        Ypred_all = -np.ones(len(e2tst))
        Ypred_all[Ipred_pos] = 1
        return Ypred_all

    def train(self, Xtrn):
        self.tune(Xtrn)
        Xrectrn = self.reconstruct(Xtrn)
        e2trn = self.calculate_loss(Xtrn, Xrectrn)
        if self.filter_moment == 1:
            e2trn = self.filter(e2trn)
        self.set_threshold(e2trn)

    def test(self, Xtst):
        Xrectst = self.reconstruct(Xtst)
        e2tst = self.calculate_loss(Xtst, Xrectst)
        if self.filter_moment == 1:
            e2tst = self.filter(e2tst)
            return self.classify(e2tst)
        else:
            Ypred = self.classify(e2tst)
            return self.filter(pd.Series(Ypred))
        