import pandas as pd
from pyedflib import highlevel
import numpy as np
import math
import matplotlib.pyplot as plt

signals, signal_headers, header = highlevel.read_edf("../data/chb01/chb01_03/chb01_03.edf")
plt.xlabel("Tempo (segundos)")
plt.ylabel("Amplitude (milivolts)")
plt.title("Sinal FP1-F7")
a = signals[0][2960*256:3100*256]
b = np.arange(len(signals[0]))/256
c = b[2960*256:3100*256]
print(len(c))
plt.plot(c, a)
plt.axvline(x=2996, color='red', linestyle='--', linewidth=1)
plt.axvline(x=3036, color='red', linestyle='--', linewidth=1)
plt.show()