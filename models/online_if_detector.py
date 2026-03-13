from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from models.encoder import encoder

class online_if_detector(encoder):

    def __init__(self, n_estimators=100, contamination=0.05, random_state=42, max_samples=256, max_features=1.0, **kwargs):
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state,
                                     max_samples=max_samples, max_features=max_features)
        self.name = 'if'
        super().__init__(**kwargs)

    def tune(self, Xtrn):
        self.model.fit(Xtrn.T)

    def reconstruct(self, X):
        return X

    def calculate_loss(self, X, Xrec):
        return -self.model.decision_function(X.T)