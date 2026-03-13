from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from models.encoder import encoder

class online_mlp_encoder(encoder):
        
    def  __init__(self, hidden_layer_sizes, activation='logistic', solver='adam', tol=1e-5, **kwargs):
        self.mlp = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                        max_iter=1000, random_state=42, early_stopping=True, tol=tol,
                        learning_rate='adaptive', batch_size=64)
        self.name = 'mlp-' + str(len(hidden_layer_sizes)) + 'camadas'
        super().__init__(**kwargs)

    def tune(self, Xtrn):
        Xtrn = Xtrn.T
        self.mlp.fit(Xtrn, Xtrn)

    def reconstruct(self, X):
        return self.mlp.predict(X.T).T