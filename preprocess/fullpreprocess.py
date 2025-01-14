from os import listdir
from preprocess import preprocess
import pickle

for name in listdir("../chb01"):
    print(name)
    dir = "chb01/" + name + "/"
    signal = preprocess(dir + name + ".edf")
    with open(dir + name + "-cov.txt", 'wb') as fp:
        pickle.dump(signal, fp)