import numpy as np
from bandpower import bandpower

freqranges = [
    (0.5, 4), # Delta
    (4, 8), # Theta
    (8, 12), # Alpha
    (13, 30) # Beta
]

def extract_features(signal):
    vet = [channel_features for channel in signal for channel_features in single_channel_extract(channel)]
    return np.array(vet).reshape(-1, 1)

def single_channel_extract(channel):
    fs = 256
    return bandpower(channel, fs, freqranges, fs, fs/2)