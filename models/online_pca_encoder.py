from models.pca import pca
import numpy as np
import pandas as pd
from models.encoder import encoder

class online_pca_encoder(encoder):

    def __init__(self, tol, **kwargs):
        self.Q = None
        self.tol = tol
        self.name = 'pca'
        super().__init__(**kwargs)
        
    def tune(self, Xtrn):
        V, VEi = pca(Xtrn)
        VE = np.cumsum(VEi)
        q = np.where(VE >= self.tol)[0][0] + 1
        Vq = V.iloc[:, :q]
        self.Q = Vq.T

    def reconstruct(self, X):
        Z = np.dot(self.Q, X)
        return np.dot(self.Q.T, Z)