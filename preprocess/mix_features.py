from os import listdir
import pickle
import numpy as np
import pandas as pd

TP = 148
FP = 12752

a  = 100.0 * TP / (TP + FP)
print(a)