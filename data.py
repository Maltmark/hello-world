"""

"""

import numpy as np
import scipy.io
import os
import numpy as np
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
    for channel in range(10):
        plt.plot(spk[channel, :, spike_ind])


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
    print(path_to_data)
    print(res_path)
    print(spk_path)
    print(clu_path)

    spk = LoadFile(spk_path)
    PlotSpike(spk, 1)
    PlotSpike(spk, 2)

    print("ASDf")







if __name__=="__main__":
    main()
