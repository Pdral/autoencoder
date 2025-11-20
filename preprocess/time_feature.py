import numpy as np
import math, nolds, time
from scipy.stats import skew, kurtosis, iqr, median_abs_deviation

features = {
    'mean': lambda signal: np.mean(signal), # Média do sinal
    'variance': lambda signal: np.var(signal), # Variância do sinal
    'skewness': lambda signal: skew(signal), # Assimetria do sinal, positiva -> direita
    'kurtosis': lambda signal: kurtosis(signal), # Curtose, se positiva indica concentração de dados nas extremidades (outliers)
    'rms': lambda signal: np.sqrt(np.mean(signal**2)), # Raiz quadrática média, estima a intensidade média do sinal
    'range': lambda signal: iqr(signal), # Range interquartil, Q3-Q1
    'mad': lambda signal: median_abs_deviation(signal), # Desvio mediano absoluto, mediana dos desvios (calculados a partir da mediana)
    'entropy': lambda signal: nolds.sampen(signal, emb_dim=2), # Entropia amostral, quanto mais alta maior a complexidade do sinal. emb_din é o tamanho a ser analisado (2 em 2)
    'aac': lambda signal: np.mean(np.abs(np.diff(signal))), # Mudança de amplitude média, se alta indica que o sinal é volátil
    'logd': lambda signal: math.exp(np.mean(np.log(np.abs(signal)))) # Log Detector, uma abordagem computacional para calcular a média geométrica
}

def extract_features(signal):
    vet = [channel_features for channel in signal for channel_features in single_channel_extract(channel)]
    return np.array(vet).reshape(-1, 1)

def single_channel_extract(channel):
    return [extraction_method(channel) for extraction_method in features.values() ]