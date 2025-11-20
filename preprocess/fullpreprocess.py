from os import listdir
from preprocess import preprocess
import pickle

root = 'data'
paths = listdir(root)
# paths = [
#     'data/chb07'
# ]

for path in paths:
    path = root + '/' + path
    for name in listdir(path):
        print(name)
        dir = path + "/" + name + "/"
        signal = preprocess(dir + name + ".edf")
        file_extension = "-freq_features.txt"
        with open(dir + name + file_extension, 'wb') as fp:
            pickle.dump(signal, fp)