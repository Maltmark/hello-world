"""

"""

import numpy as np
import scipy.io
import os
import numpy as np
# from peakdetect import peakdetect
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def LoadFile(path):
    file_name = (os.path.splitext(os.path.basename(path))[0])
    print(file_name)
    return scipy.io.loadmat(path)[file_name]

def PlotSpike(spk,spike_ind):
    plt.figure()
    plt.xlabel('---WHAT IS THIS AXIS---')
    plt.ylabel('spiking rate in mv')
    plt.title('Spiking waveform #{}'.format(spike_ind))
    plt.grid(True)
    # for channel in range(10):
    #     plt.plot(spk[channel, :, spike_ind])
    plt.plot(spk[1, :, spike_ind])
    x = - spk[1, :, spike_ind]
    peaks, _ = find_peaks(x)
    # plt.plot(x)
    plt.plot(np.argmin(x[peaks]),  - np.min(x[peaks]), "x")
    # plt.plot(np.zeros_like(x), "--", color="gray")
    plt.show()




def main():
    print """
===================================================
Brain proj
===================================================
* Michal Altmark  id-  mail-
* David Gertskin  id-315003947  mail-David.gertskin@gmail.com
-------------------------------------------------------------------
Subject : we want to handle the data loading and parsing
-------------------------------------------------------------------
"""

    path_to_data = "C:\Users\David Gertskin\Desktop\school\BrainProj\DATA\FirstDataset"
    clu_path = path_to_data + r"\clu.mat"
    res_path = path_to_data + r"\res.mat"
    spk_path = path_to_data + r"\spk.mat"


    spk = LoadFile(spk_path)
    PlotSpike(spk, 1)
    PlotSpike(spk, 2)









if __name__=="__main__":
    main()
