import numpy as np
from bandpower import bandpower
from preprocess.feature_extractor import FeatureExtractor

class FreqFeatureExtractor(FeatureExtractor):
    freqranges = [
        (0.5, 4), # Delta
        (4, 8), # Theta
        (8, 12), # Alpha
        (13, 30) # Beta
    ]

    def extract_features(self, signal):
        vet = [channel_features for channel in signal for channel_features in self.single_channel_extract(channel)]
        return np.array(vet).reshape(-1, 1)

    def single_channel_extract(self, channel):
        fs = 256
        return bandpower(channel, fs, self.freqranges, fs, fs/2)