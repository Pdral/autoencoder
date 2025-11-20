import math

import numpy as np
from math import floor
import pickle
from pathlib import Path
import os

def writeLabels(start_hour, start_min, start_sec, end_hour, end_min, end_sec, seizures, dir, name):
    size = (end_hour-start_hour)*3600 + (end_min-start_min)*60 + (end_sec-start_sec)
    E = math.ceil(size/2)
    Y = -np.ones(E)
    for start_seizure, end_seizure in seizures:
        Y[range(floor(start_seizure/2), floor((end_seizure/2)+0.5))] = 1
    Path(dir).mkdir(parents=True, exist_ok=True)
    with open(dir + name, 'wb') as fp:
        pickle.dump(Y, fp)
    os.rename("../data/chb07/chb07_" + n + ".edf", dir + "chb07_" + n + ".edf")

if __name__=="__main__":
    start_hour = 11
    start_min = 46
    start_sec = 29
    end_hour = 12
    end_min = 48
    end_sec = 35
    seizures = [(3285,3381)]
    n = "13"
    dir = "../data/chb07/chb07_" + n + "/"
    name = "chb07_" + n + "-labels.txt"
    writeLabels(start_hour, start_min, start_sec, end_hour, end_min, end_sec, seizures, dir, name)