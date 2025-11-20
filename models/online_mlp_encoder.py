from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd

class online_mlp_encoder():
        
    def  __init__(self, hidden_layer_sizes, activation='logistic', solver='adam', tol=1e-5):
        self.mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                        max_iter=1000, random_state=42, early_stopping=True, tol=tol,
                        learning_rate='adaptive', batch_size=64)
        self.L = None
        self.name = 'mlp-' + str(len(hidden_layer_sizes)) + 'camadas'

    def train(self, Xtrn, perc=95):
        Xtrn = Xtrn.T
        self.mlp.fit(Xtrn, Xtrn)
        Xrectrn = self.mlp.predict(Xtrn)
        Etrn = (Xtrn - Xrectrn).T
        e2trn = []
        for i in range(Xtrn.shape[0]):
            e2trn.append(np.dot(Etrn.iloc[:, i].T, Etrn.iloc[:, i]))
        self.L = np.percentile(e2trn, perc)

    def test(self, Xtst):
        Xtst = Xtst.T
        Xrectst = self.mlp.predict(Xtst)
        Etst = pd.DataFrame((Xtst - Xrectst).T)
        e2tst = []
        for i in range(Xtst.shape[0]):
            e2tst.append(np.dot(Etst.iloc[:, i].T, Etst.iloc[:, i]))
        Ipred_pos = list(np.where(e2tst > self.L)[0])
        Ypred_all = -np.ones(Xtst.shape[0])
        Ypred_all[Ipred_pos] = 1
        return Ypred_all