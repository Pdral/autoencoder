from models.pca import pca
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class detector(ABC):

    def __init__(self):
        self.name = 'general-detector'

    @abstractmethod
    def train(self, Xtrn):
        pass

    @abstractmethod
    def test(self, Xtst):
        pass