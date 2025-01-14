import math

import numpy as np
from math import floor
import pickle
from pathlib import Path

def writeLabels(start_hour, start_min, start_sec, end_hour, end_min, end_sec, seizures, dir, name):
    size = (end_hour-start_hour)*3600 + (end_min-start_min)*60 + (end_sec-start_sec)
    E = math.ceil(size/2)
    Y = -np.ones(E)
    for start_seizure, end_seizure in seizures:
        Y[range(floor(start_seizure/2), floor((end_seizure/2)+0.5))] = 1
    Path(dir).mkdir(parents=True, exist_ok=True)
    with open(dir + name, 'wb') as fp:
        pickle.dump(Y, fp)

if __name__=="__main__":
    start_hour = 13
    start_min = 43
    start_sec = 4
    end_hour = 14
    end_min = 43
    end_sec = 4
    seizures = [(2996, 3036)]
    n = "03"
    dir = "chb01/chb01_" + n + "/"
    name = "chb01_" + n + "-labels.txt"
    writeLabels(start_hour, start_min, start_sec, end_hour, end_min, end_sec, seizures, dir, name)