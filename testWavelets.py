import sys
import datetime
import random
import pickle
from os.path import isfile, join
from os import listdir

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
from scipy.io import loadmat

import pywt

def getDataLPBA40(fileName, fileName2):
    dtype = np.dtype('<u2')
    fid = open(fileName, 'rb')
    xData = np.fromfile(fid, dtype)
    fid.close()
    xData = xData.reshape(256, -1, 256)
    xData = xData/np.max(xData)
    wanted = xData[:, 50, :]
    wanted = np.flip(wanted, 0)

    label_ = nib.load(fileName2).get_data()
    label_ = label_.reshape(256, -1, 256)
    label_ = np.where(label_!=0, 1, 0)
    wantedl = label_[:, 50, :]
    wantedl = np.flip(np.swapaxes(wantedl, 0, 1), 0)

    plt.subplot(121)
    plt.imshow(wanted, cmap='gray')
    plt.subplot(122)
    #plt.imshow(wantedl, cmap='gray')
    plt.imshow(wanted, cmap='gray')
    plt.imshow(wantedl, cmap='gray', alpha=0.5)
    plt.show()
    cA, (cH, cV, cD) = pywt.dwt2(wanted, 'haar', mode=2)
    print(np.max(cH+cV+cD))
    plt.subplot(231)
    plt.imshow(cA, cmap='gray')
    plt.imshow(scipy.misc.imresize(wantedl, 0.5), cmap='Blues', alpha=0.1)
    
    plt.subplot(232)
    plt.imshow(cH, cmap='gray')
    
    plt.subplot(233)
    plt.imshow(cV, cmap='gray')
    plt.subplot(234)
    plt.imshow(cD, cmap='gray')
    plt.subplot(235)
    plt.imshow(cH+cV, cmap='gray')
    plt.imshow(scipy.misc.imresize(wantedl, 0.5), cmap='Blues', alpha=0.3)
    plt.subplot(236)
    plt.imshow(cH+cV+cD, cmap='gray')
    plt.show()
 

getDataLPBA40('../lpba/lpba_img/S01.native.mri.img', '../lpba/lpba_img/S01.native.tissue.img')

