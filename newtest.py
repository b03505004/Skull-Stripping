import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import nibabel as nib
import scipy.misc
from os.path import isfile, join
from os import listdir
from skimage.filters import threshold_otsu
import sys
from scipy.io import loadmat

np.set_printoptions(threshold=256*256)

dataset = sys.argv[1]
#zs = [1,3,5,7]
#zs = [1, 3, 5]
zs = [1, 3, 5]
z_halfs = [int(i/2) for i in zs]

print("Zs:", zs)

label_val = 1

def getDataIBSR20(fileName, z, z_half, original):
    xData = loadmat(fileName)['volume']
    temp = np.zeros((256, 256, xData.shape[2]+2*z_half))
    xData = xData - np.min(xData)
    temp[..., z_half:temp.shape[2]-z_half] = xData
    plt.imshow(temp[...,50], cmap='gray')
    plt.show()
    #print(temp.shape)
    for i in range(temp.shape[2]):
        original.append(temp[..., i])

def getLabelIBSR20(fileName, z, z_half, label):
    label_ = loadmat(fileName)['gt'] # test ibsr
    temp = np.where(label_!=0, label_val, 0)

    for i in range(label_.shape[2]):
        label.append(temp[..., i].reshape(256, 256, 1))

def getDataLPBA40(fileName, z, z_half, original):
    dtype = np.dtype('<u2')
    fid = open(fileName, 'rb')
    xData = np.fromfile(fid, dtype)
    fid.close()
    xData = xData.reshape(256, -1, 256)

    temp = np.zeros((256, xData.shape[1]+2*z_half, 256))
    temp[:, z_half:temp.shape[1]-z_half, :] = xData
    for j in range(100):
        plt.imshow(np.flip(temp[:, 10+j, :], 0), cmap='gray')
        plt.show()
    #print(temp.shape)
    for i in range(temp.shape[1]):
        original.append(np.flip(temp[:, i, :], 0))

def getLabelLPBA40(fileName, z, z_half, label):
    label_ = nib.load(fileName).get_data()
    label_ = label_.reshape(256, -1, 256)
    label_ = np.where(label_!=0, label_val, 0)
    temp = np.zeros((256, label_.shape[1]+2*z_half, 256))
    temp[:, z_half:temp.shape[1]-z_half, :] = label_

    for i in range(temp.shape[1]):
        label.append(np.where(np.flip(np.swapaxes(temp[:, i, :], 0,1), 0)!=0, label_val, 0).reshape(256, 256, 1))


l = []
getDataIBSR20('../ibsr/20mat/IBSR01_1', 1, 0, l)
getDataLPBA40('../lpba/lpba_img/S40.native.mri.img',1, 0, l)