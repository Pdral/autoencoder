import numpy as np
from scipy import signal
from scipy.integrate import trapezoid

def bandpower(x, fs, freqrange, nperseg=1024, noverlap=None):
    frequencies, psd = signal.welch(x, fs, nperseg=nperseg, noverlap=noverlap)
    p = []
    for lower, higher in freqrange:
        freq_indices = np.where((frequencies >= lower) & (frequencies <= higher))
        p.append(trapezoid(psd[freq_indices], frequencies[freq_indices]))
    return p