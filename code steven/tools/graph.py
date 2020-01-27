import numpy as np
import matplotlib.pyplot as plt
import os

def grapher():
    path = '/media/public_data/aeroscan/dataset/paint_dataset/fx10+gt/profiels'
    profile = ['hsprofile_trimetal', 'hsprofile_sikkens','hsprofile_sigma','hsprofile_wyzonol']
    channels = [c for c in range(224)]
    pixvals = list()


    for name in profile:
        loaded = np.load(os.path.join(path, name + '.npy'))
        pixvals.append(loaded[0])
    plt.plot(channels, pixvals[0], '-y', label="Trimetal")
    plt.plot(channels, pixvals[1], '-g', label="Sikkens")
    plt.plot(channels, pixvals[2], '-r', label="Sigma")
    plt.plot(channels, pixvals[3], '-b', label="Wyzonol")
    plt.legend(loc="upper right", prop={'size':12})
    plt.title("Hyper-spectral profiles of the 9001 paint from different manufacturers (fx10)")
    plt.xlabel("Channels")
    plt.ylabel("Pixel value")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    grapher()