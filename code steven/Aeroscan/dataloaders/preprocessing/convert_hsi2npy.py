from spectral import imshow, view_cube
import spectral.io.envi as envi
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import os


def hc2hhsi(hc):
    print("Transforming n-dim hypercube to hyper-hue, saturation and intensity...")
    # Transform an n-dimensional hypercube to hyper-hue, saturation and intensity.
    ####################################################################################################################
    # Calculate the components c
    rows = hc.shape[0]
    cols = hc.shape[1]
    dims = hc.shape[2]

    c = np.zeros((rows, cols, dims-1))
    for i in range(dims - 1):
        nonZeroEle = dims - i # nonZeroEle is the number of non-zero elements of the base unit vector u1, u2, ...
        c[:, :, i] = (nonZeroEle - 1) ** 0.5 / nonZeroEle ** 0.5         * hc[:, :, i] \
                     - 1 / ((nonZeroEle - 1) ** 0.5 * nonZeroEle ** 0.5) * np.sum(hc[:, :, i+1:dims], axis=2)
    ####################################################################################################################

    # Normalise the norms of c to 1 to obtain hyper-hue hh.
    c_norm = np.sum(c ** 2, axis=2) ** 0.5
    c_norm = c_norm + (c_norm == 0) * 1e-10
    c_norm = np.tile(c_norm, (dims - 1, 1, 1))
    c_norm = np.moveaxis(c_norm, 0, -1)
    hh = c / c_norm # add 1e-10 to avoid zero denominators
    hues = torch.as_tensor(np.arctan2(hh[..., 1:], hh[..., :-1]),dtype = torch.float32)

    return hues
    # -------------------------------------------------------------------------------

def flat_field(input, dark_ref, white_ref):
    ffc = np.divide(np.subtract(input, dark_ref),
                    np.subtract(white_ref, dark_ref))
    return ffc

def linescan_balancechannels_bg(cube: np.ndarray, bg_width: int = 30) -> np.ndarray:
    # author: Martin Dijkstra
    cube_bg = cube[:, :, 0:bg_width]
    chan_avg = np.average(cube_bg, axis=(1, 2))
    gain = np.max(chan_avg) / chan_avg
    gain = np.lib.stride_tricks.as_strided(gain, shape=cube.shape, strides=(gain.strides[0], 0, 0), writeable=False)
    return cube * gain

def bg_balance(ffc):
    ffc_bg = np.transpose(linescan_balancechannels_bg(np.transpose(ffc,(2,0,1))),(1,2,0))
    return ffc_bg

def grad_log(ffc_bg):
    ffc_bg_grad = np.gradient(ffc_bg, axis=2)  # gradient uses central difference
    ffc_bg_grad_log = np.divide(ffc_bg_grad, ffc_bg+1e-10)
    return ffc_bg_grad_log

def hsi2npy(img_name, data_file, output_file):

    print(">>> Opening HSI...")
    input_path = os.path.join(data_file, img_name,'capture', img_name)
    dark_path = os.path.join(data_file, img_name,'capture', 'DARKREF_' + img_name)
    white_path = os.path.join(data_file, img_name,'capture', 'WHITEREF_' + img_name)

    hsi_input = envi.open(input_path +'.hdr',input_path + '.raw')
    hsi_dark = envi.open(dark_path+'.hdr',dark_path + '.raw')
    hsi_white = envi.open(white_path + '.hdr', white_path + '.raw')

    print(">>> Loading into numpy arrays...")
    # Initial shapes HxWxC: 1091x640x224, 200x640x224, 152x640x224
    npy_input = np.array(hsi_input.load())
    npy_dark = np.array(hsi_dark.load())
    npy_white = np.array(hsi_white.load())

    print(">>>>>> Resizing the references...")
    dark_plane = np.full(npy_input.shape,np.mean(npy_dark, axis=0))
    white_plane = np.full(npy_input.shape,np.mean(npy_white, axis=0))
    npy_dark,npy_white = None, None

    print(">>> Correcting using dark and white references...")
    # Pixels values e [0,1], proportionnal to black and white refs -- Flat field correction
    ffc_input= flat_field(npy_input,dark_plane, white_plane)
    npy_input, dark_plane,white_plane = None, None, None

    # Logarithmic derivative to do
    #ffc_bg_input = bg_balance(ffc_input)
    #corrected_input = grad_log(ffc_bg_input)

    #hyper-hues
    corrected_input=hc2hhsi(ffc_input)

    print(">>> Saving to", output_file)
    np.save(output_file, corrected_input)

    print(">>> Done\n")

# Loop for all the imgs

with open("/home/student/aeroscan/tileimp/Aeroscan/dataloaders/preprocessing/preprocess.txt", 'r') as file:
    lines = file.read().splitlines()
    for line in lines:
        if line == 'FX17':
            print("*****Pre-processing FX17 data*****")
            data_file = '/media/public_data/aeroscan/dataset/paint_dataset/fx17+gt'
            continue
        elif line == 'FX10':
            print("*****Pre-processing FX10 data*****")
            data_file = '/media/public_data/aeroscan/dataset/paint_dataset/fx10_doublescan'
            continue
        print("To be pre-processed", line)
        output_file = os.path.join('/media/public_data/aeroscan/dataset/paint_dataset/fx10_doublescan/alldata_hue', line + '.npy')
        hsi2npy(line, data_file, output_file)
    print("All files saved.")