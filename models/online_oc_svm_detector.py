from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
from models.encoder import encoder

class online_oc_svm_detector(encoder):

    def __init__(self, kernel='rbf', gamma='auto', nu=0.05, **kwargs):
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.name = 'oc-svm'
        super().__init__(**kwargs)

    def tune(self, Xtrn):
        self.model.fit(Xtrn.T)

    def reconstruct(self, X):
        return X

    def calculate_loss(self, X, Xrec):
        return -self.model.decision_function(X.T)