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
import random
import pickle
from scipy.io import loadmat
import datetime


random.seed(8787)

dataset = sys.argv[3]

np.set_printoptions(threshold=256*256)
z = int(sys.argv[1])
z_half = int(z/2)
label_val = 255
print('Z:', z)

def shuffle(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a,b

train_label = []
train_x = []
val_label = []
val_x = []


labelMinc = []
originalMinc = []

labelIBSR = []
originalIBSR = []

label_val = 1

def getDataMinc(fileName):
    xData = nib.load(fileName).get_data()
    temp = np.zeros((256,256+2*z_half,256))
    xData = xData - np.min(xData)
    temp[60:181+60, z_half:temp.shape[1]-z_half, :] = xData
    #xData = temp

    for i in range(temp.shape[1]):
        originalMinc.append(np.flip(temp[:, i, :], 0))

def getLabelMinc(fileName):
    label_ = nib.load(fileName).get_data()
    label_ = np.where(((label_==1)|(label_==2)|(label_==3)), 1, 0)
    temp = np.zeros((362,512,512))
    temp[..., 40:512-38, 76:512-74] = label_
    
    temp2 = np.zeros((181,256,256))
    for i in range(temp2.shape[1]):
        temp2[:, i, :] = np.where(scipy.misc.imresize(temp[:,2*i,:], 0.5, interp='nearest')!=0, label_val, 0)


    temp3 = np.zeros((256,256+2*z_half,256))
    temp3[60:181+60, z_half:temp3.shape[1]-z_half, :] = temp2
    
    for i in range(temp3.shape[1]):
        labelMinc.append(np.flip(temp3[:, i, :].reshape(256, 256, 1), 0))

def getDataIBSR(fileName):
    xData = nib.load(fileName).get_data()
    xData = xData - np.min(xData)

    temp = np.zeros((xData.shape[0]+2*z_half, 256, 256))
    temp[z_half:temp.shape[0]-z_half, ...] = xData
    #xData = temp

    for i in range(temp.shape[0]):
        originalIBSR.append(temp[i, ...])

def getLabelIBSR(fileName):
    label_ = np.where(nib.load(fileName).get_data()!=0 , label_val, 0)

    temp = np.zeros((128+2*z_half, 256, 256))
    temp[z_half:temp.shape[0]-z_half, ...] = label_

    for i in range(temp.shape[0]):
        labelIBSR.append(temp[i, ...].reshape(256, 256, 1))

def getDataIBSR20TC(fileName, z, z_half, original):
    xData = np.load(fileName+'.tam_con.npy')
    xData = xData/np.max(xData)

    temp = np.zeros((256, 256, xData.shape[2]+2*z_half))
    temp[..., z_half:temp.shape[2]-z_half] = xData
    for i in range(temp.shape[2]):
        original.append(temp[..., i])

def getDataIBSR20(fileName, z, z_half, original):
    xData = loadmat(fileName)['volume']
    xData = xData - np.min(xData)
    xData = xData/np.max(xData)
    temp = np.zeros((256, 256, xData.shape[2]+2*z_half))
    temp[..., z_half:temp.shape[2]-z_half] = xData
    #print(temp.shape)
    for i in range(temp.shape[2]):
        original.append(temp[..., i])

def getLabelIBSR20(fileName, z, z_half, label):
    label_ = loadmat(fileName)['gt'] # test ibsr
    label_ = np.where(label_!=0, label_val, 0)
    temp = np.zeros((256,256,label_.shape[2]+2*z_half))
    temp[..., z_half:temp.shape[2]-z_half] = label_

    for i in range(temp.shape[2]):
        label.append(temp[..., i].reshape(256, 256, 1))

def getDataLPBA40TC(fileName, z, z_half, original):
    xData = np.load(fileName+'.tam_con.npy')
    xData = xData/np.max(xData)

    temp = np.zeros((256, xData.shape[1]+2*z_half, 256))
    temp[:, z_half:temp.shape[1]-z_half, :] = xData
    #print(temp.shape)
    for i in range(temp.shape[1]):
        original.append(np.flip(temp[:, i, :], 0))
def getDataLPBA40(fileName, z, z_half, original):
    dtype = np.dtype('<u2')
    fid = open(fileName, 'rb')
    xData = np.fromfile(fid, dtype)
    fid.close()
    xData = xData.reshape(256, -1, 256)
    xData = xData/np.max(xData)

    temp = np.zeros((256, xData.shape[1]+2*z_half, 256))
    temp[:, z_half:temp.shape[1]-z_half, :] = xData
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

"""minc20Files = sorted([f for f in listdir('../minc/20/') if isfile(join('../minc/20/', f))])
#print(minc20Files)
minc20Files = minc20Files[1:]
#minc20Files = minc20Files[:4]
#print(minc20Files)
for i, mFile in enumerate(minc20Files):
    if mFile.find('crisp')!=-1:
        getLabelMinc('../minc/20/'+mFile)
    elif mFile.find('t1w')!=-1:
        getDataMinc('../minc/20/'+mFile)
labelMinc = np.array(labelMinc)
originalMinc = np.array(originalMinc)
print("BrainWeb:", originalMinc.shape, labelMinc.shape)

bn = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

for b in bn:
    filePathX = '../ibsr/10/IBSR_'+b+'/images/MINC/IBSR_'+b+'_ana.hdr.mnc'
    filePathSeg = '../ibsr/10/IBSR_'+b+'/segmentation/MINC/IBSR_'+b+'_segTRI_ana.hdr.mnc'
    getDataIBSR(filePathX)
    getLabelIBSR(filePathSeg)

labelIBSR = np.array(labelIBSR)
originalIBSR = np.array(originalIBSR)
print("IBSR:", originalIBSR.shape, labelIBSR.shape)
"""
train_x = []
train_xtc = []
train_label = []

val_x = []
val_xtc = []
val_label = []

#val_x2 = []
#val_label2 = []
"""
def getTorchX():
    for i in range(originalMinc.shape[0]):
        mi = i%(256+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(256+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_x.append(torch.from_numpy(np.reshape(originalMinc[i-z_half : i+z_half+1,...], (1,1,z,originalMinc.shape[1],256))).float())
                #plt.imsave(arr=original[i, ...], cmap='gray', fname='./all/'+str(i)+'_x.jpg')
            else:
                train_x.append(torch.from_numpy(np.reshape(originalMinc[i-z_half : i+z_half+1,...], (1,1,z,originalMinc.shape[1],256))).float())
                #plt.imsave(arr=original[i, ...], cmap='gray', fname='./all/'+str(i)+'_x.jpg')

def getTorchLabel():
    for i in range(originalMinc.shape[0]):
        mi = i%(256+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(256+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_label.append(labelMinc[i, ...])
                #plt.imsave(arr=label[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(i)+'_lab.jpg')
            else:
                train_label.append(torch.from_numpy(np.reshape(labelMinc[i, ...], (1, 1, 1, labelMinc.shape[1], 256))).float())
                #plt.imsave(arr=label[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(i)+'_lab.jpg')
"""
def getTorchXIBSR():
    for i in range(originalIBSR.shape[0]):
        mi = i%(128+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(128+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_x.append(torch.from_numpy(np.reshape(originalIBSR[i-z_half : i+z_half+1,...], (1,1,z,originalIBSR.shape[1],256))).float())
                #plt.imsave(arr=originalIBSR[i, ...], cmap='gray', fname='./all/'+str(181*20+i)+'_x.jpg')
            else:
                train_x.append(torch.from_numpy(np.reshape(originalIBSR[i-z_half : i+z_half+1,...], (1,1,z,originalIBSR.shape[1],256))).float())
                #plt.imsave(arr=originalIBSR[i, ...], cmap='gray', fname='./all/'+str(181*20+i)+'_x.jpg')

            
def getTorchLabelIBSR():
    for i in range(labelIBSR.shape[0]):
        mi = i%(128+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(128+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_label.append(labelIBSR[i, ...])
                #plt.imsave(arr=labelIBSR[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(181*20+i)+'_lab.jpg')
            else:
                train_label.append(torch.from_numpy(np.reshape(labelIBSR[i, ...], (1, 1, 1, labelIBSR.shape[1], 256))).float())
                #plt.imsave(arr=labelIBSR[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(181*20+i)+'_lab.jpg')

def getTorchX(z, z_half, original, isVal, train_x, val_x):
    if isVal:
        val_x.append([])
    for i in range(original.shape[0]):
        mi = i%(original.shape[0])
        if ((mi-z_half)<0) or ((mi+z_half)>=(original.shape[0])):
            continue
        else:
            if (isVal):
                val_x[-1].append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,original.shape[1],256))).float())
                #plt.imsave(arr=original[i, ...], cmap='gray', fname='./all/'+str(181*20+i)+'_x.jpg')
            else:
                train_x.append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,original.shape[1],256))).float())
                #plt.imsave(arr=original[i, ...], cmap='gray', fname='./all/'+str(181*20+i)+'_x.jpg')

def getTorchLabel(z, z_half, label, isVal, train_label, val_label):
    if isVal:
        val_label.append([])
    for i in range(label.shape[0]):
        mi = i%(label.shape[0])
        if ((mi-z_half)<0) or ((mi+z_half)>=(label.shape[0])):
            continue
        else:
            if (isVal):
                val_label[-1].append(label[i, ...])
                #plt.imsave(arr=label[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(181*20+i)+'_lab.jpg')
            else:
                train_label.append(torch.from_numpy(np.reshape(label[i, ...], (1, 1, 1, label.shape[1], 256))).float())
                #plt.imsave(arr=label[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(181*20+i)+'_lab.jpg')

print("________________________________________________________________________")
"""getTorchX()
getTorchLabel()
getTorchLabelIBSR()
getTorchXIBSR()
print('Train label:', len(train_label), train_label[0].shape)
print('Train x:', len(train_x), train_x[0].shape)
print('Val label:', len(val_label), val_label[0].shape)
print('Val x:', len(val_x), val_x[0].shape)"""

k = int(sys.argv[2])
print("K:", k)
if dataset=="ibsr20":
    bn = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    """for b in bn:
        filePathX = '../ibsr/10/IBSR_'+b+'/images/MINC/IBSR_'+b+'_ana.hdr.mnc'
        filePathSeg = '../ibsr/10/IBSR_'+b+'/segmentation/MINC/IBSR_'+b+'_segTRI_ana.hdr.mnc'
        getDataIBSR(filePathX)
        getLabelIBSR(filePathSeg)

    labelIBSR = np.array(labelIBSR)
    originalIBSR = np.array(originalIBSR)
    print("IBSR:", originalIBSR.shape, labelIBSR.shape)
    getTorchLabelIBSR()
    getTorchXIBSR()
    print('Train label:', len(train_label), train_label[0].shape)
    print('Train x:', len(train_x), train_x[0].shape)
    print('Val label:', len(val_label), val_label[0].shape)
    print('Val x:', len(val_x), val_x[0].shape)"""
    KK = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

    to_fold = [KK[a] for a in range(len(KK)) if a not in range(k*4,k*4+4)]
    to_val = [random.choice(to_fold)]
    tempv = list(to_fold)
    tempv.remove(to_val[0])
    to_val.append(random.choice(tempv))
    print(to_val)
    for brain_num in to_fold:
        if brain_num not in to_val:
            isVal = False
        else:
            print(brain_num)
            isVal = True
        label1 = []
        original1 = []
        getDataIBSR20('../ibsr/20mat_1_gt/IBSR'+brain_num+'_1', z, z_half, original1)
        getLabelIBSR20('../ibsr/20mat_1_gt/IBSR'+brain_num+'_gt', z, z_half, label1)
        label1 = np.array(label1)
        original1 = np.array(original1)
        print(brain_num)
        print(original1.shape, label1.shape)
        getTorchX(z, z_half, original1, isVal)
        getTorchLabel(z, z_half, label1, isVal)
        #print(len(val_label), val_label[0].shape)
        print(len(train_label), train_label[0].shape)
        #print(len(val_x), val_x[0].shape)
        print(len(train_x), train_x[0].shape)
        print("______________________________________")

if dataset=='lpba':
#**********************************LPBA**********************************

    ind = [str(i)+str(j) for i in range(4) for j in range(10)][1:]
    ind.append("40")
    print(ind)
    #to_fold = [ind[a] for a in range(len(ind)) if a not in range(k*8,k*8+8)]
    if k == 0:
        to_val = ind[20:]
    else:
        to_val = ind[:20]
    """tempv = list(to_fold)
    for i in range(2):
        tempv.remove(to_val[i])
        to_val.append(random.choice(tempv))
    print(to_val)"""
    for brain_num in ind:
        if brain_num not in to_val:
            isVal = False
        else:
            print(brain_num)
            isVal = True
        label1 = []
        original1 = []
        originalTC = []
        getDataLPBA40('../lpba/lpba_img/S'+brain_num+'.native.mri.img', z, z_half, original1)
        getLabelLPBA40('../lpba/lpba_img/S'+brain_num+'.native.tissue.img', z, z_half, label1)
        #getDataLPBA40TC('../lpba/lpba_img/S'+brain_num+'.native.mri.img', z, z_half, originalTC)
        label1 = np.array(label1)
        original1 = np.array(original1)
        originalTC = np.array(originalTC)
        print(brain_num)
        print(original1.shape, originalTC.shape, label1.shape)
        getTorchX(z, z_half, original1, isVal, train_x, val_x)
        #getTorchX(z, z_half, originalTC, isVal, train_xtc, val_xtc)
        getTorchLabel(z, z_half, label1, isVal, train_label, val_label)
        #print(len(val_label), val_label[0].shape)
        print(len(train_label), train_label[0].shape)
        #print(len(val_x), val_x[0].shape)
        print(len(train_x), train_x[0].shape)
        print("______________________________________")



shuffle(train_label, train_x)
training_size = len(train_label)
print('Train label:', len(train_label), train_label[0].shape)
print('Train x:', len(train_x), len(train_xtc), train_x[0].shape)
#print('Val label:', len(val_label), val_label[0].shape)
#print('Val x:', len(val_x), val_x[0].shape)
print('Val x:', len(val_x), len(val_xtc), len(val_x[0]))

class fcn(nn.Module):
    def __init__(self):
        super(fcn, self).__init__()

        # ------------------------------for main X------------------------------
        #self.max_pool = nn.MaxPool3d(kernel_size=(3,1,1), stride=(1,1,1))
        #original MRI part
        self.c0 = nn.Conv3d(1, 16, kernel_size=(z,1,1), stride=(1,1,1))
        self.bn_0     = nn.BatchNorm3d(16)
        self.c02 = nn.Conv3d(16, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_02     = nn.BatchNorm3d(32)
        self.conv_1 = nn.Conv3d(32, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_1     = nn.BatchNorm3d(32)

        #tamura contrast part
        self.c0tc = nn.Conv3d(1, 16, kernel_size=(z,1,1), stride=(1,1,1))
        self.bn_0tc     = nn.BatchNorm3d(16)
        self.c02tc = nn.Conv3d(16, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_02tc     = nn.BatchNorm3d(32)
        self.conv_1tc = nn.Conv3d(32, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_1tc     = nn.BatchNorm3d(32)

        #fuse part
        self.conv_2 = nn.Conv3d(32, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_2     = nn.BatchNorm3d(64)
        self.conv_3 = nn.Conv3d(64, 128, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_3     = nn.BatchNorm3d(128)

        self.upsample_1 = nn.ConvTranspose3d(128, 64, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_5 = nn.BatchNorm2d(64)
        self.upsample_2 = nn.ConvTranspose3d(64, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_6     = nn.BatchNorm2d(32)
        self.upsample_3 = nn.ConvTranspose3d(32, 16, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_7     = nn.BatchNorm2d(16)
        self.upsample_4 = nn.ConvTranspose3d(16, 1, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_8     = nn.BatchNorm2d(1)
        # ------------------------------for main X------------------------------
    #def forward(self, x, tc):
    def forward(self, x):

        x = F.dropout3d(self.bn_0(F.relu(self.c0(x))), p=0.1)
        x = F.dropout3d(self.bn_02(F.relu(self.c02(x))), p=0.2)
        x = F.dropout3d(self.bn_1(F.relu(self.conv_1(x))), p=0.2)
        
        """tc = F.dropout3d(self.bn_0tc(F.relu(self.c0tc(tc))), p=0.1)
        tc = F.dropout3d(self.bn_02tc(F.relu(self.c02tc(tc))), p=0.2)
        tc = F.dropout3d(self.bn_1tc(F.relu(self.conv_1tc(tc))), p=0.2)"""
        
        #x = x + tc

        s1 = x
        x = F.dropout3d(self.bn_2(F.relu(self.conv_2(x))), p=0.3)
        s2 = x
        x = F.dropout3d(self.bn_3(F.relu(self.conv_3(x))), p=0.3)
        s3 = x
        #x = F.dropout2d(self.bn_4(F.relu(self.conv_4(x))), p=0.3)
        
        s3 = F.dropout3d(self.bn_5(F.relu(self.upsample_1(s3))), p=0.3)
        s3 = s3 + s2
        s3 = F.dropout3d(self.bn_6(F.relu(self.upsample_2(s3))), p=0.3)
        s3 = s3 + s1
        s3 = F.dropout3d(self.bn_7(F.relu(self.upsample_3(s3))), p=0.3)
        s3 = F.dropout3d(self.bn_8(F.sigmoid(self.upsample_4(s3))), p=0.2)
        return s3
        
def val():
    total = 0
    union = 0
    for i in range(len(val_label)):
        val_input = Variable(val_x[i])
        out = net(val_input).data.numpy()
        out = np.reshape(out,(val_x[i].numpy().shape[3],256))
        try:
            thresh = threshold_otsu(out)
        except:
            thresh = 255
        #print(thresh)
        plt.imsave(arr=val_x[i].numpy().reshape(z,-1,256)[z_half,...], cmap='gray', fname='./plot/'+str(i)+'_'+'input.jpg')
        plt.imsave(arr=out, cmap='gray', fname='./plot/'+str(i)+'_'+'b4.jpg')
        out = np.reshape(np.where(out>=thresh, 1, 0), (-1,256))
        #print(out)
        plt.imsave(arr=out, cmap='gray', fname='./plot/'+str(i)+'_'+'.jpg')
        lab = np.reshape(np.where(val_label[i]==label_val, 1, 0), (-1,256))
        plt.imsave(arr=np.reshape(lab,(-1,256)), cmap='gray', fname='./plot/'+str(i)+'_'+'lab.jpg')
        total += np.sum(out) + np.sum(lab)
        temp = out+lab
        #print(temp)
        union += np.sum(np.where(temp==2, 1, 0))
    if total==0:
        return 0
    else:
        return 2*union/total


def val2(net, val_label, val_x, val_xtc):
    dicecoef_sum = 0
    jaccard_sum = 0
    for b,brain in enumerate(val_label):
        total = 0
        intersect = 0

        jtotal = 0
        jintersect = 0
        for i in range(len(brain)):
            val_input = Variable(val_x[b][i])
            #val_inputtc = Variable(val_xtc[b][i])
            #out = net(val_input, val_inputtc).data.numpy()
            out = net(val_input).data.numpy()
            out = np.reshape(out,(val_x[b][i].numpy().shape[3],256))

            #print(thresh)
            plt.imsave(arr=val_x[b][i].numpy().reshape(z,-1,256)[z_half,...], cmap='gray', fname='./plot/v2_'+str(i)+'_'+'input.jpg')
            #plt.imsave(arr=val_xtc[b][i].numpy().reshape(z,-1,256)[z_half,...], cmap='gray', fname='./plot/v2_'+str(i)+'_'+'inputtc.jpg')
            plt.imsave(arr=out, cmap='gray', fname='./plot/v2_'+str(i)+'_'+'b4.jpg')
            out = np.reshape(np.where(out>=0.5, 1, 0), (-1,256))
            #print(out)
            plt.imsave(arr=out, cmap='gray', fname='./plot/v2_'+str(i)+'_'+'.jpg')
            lab = np.reshape(np.where(brain[i]==label_val, 1, 0), (-1,256))
            plt.imsave(arr=np.reshape(lab,(-1,256)), cmap='gray', fname='./plot/v2_'+str(i)+'_'+'lab.jpg')
            total += np.sum(out) + np.sum(lab)
            temp = out+lab
            #print(temp)
            intersect += np.sum(np.where(temp==2, 1, 0))
            
            jtotal += np.sum(out) + np.sum(lab) - np.sum(np.where(temp==2, 1, 0))
            jintersect += np.sum(np.where(temp==2, 1, 0))
        print(2*intersect/total, jintersect/jtotal)
        dicecoef_sum += 2*intersect/total
        jaccard_sum += jintersect/jtotal
    print("AVG:", dicecoef_sum/len(val_label), jaccard_sum/len(val_label))
    return dicecoef_sum/len(val_label)
batch_size = 6
epochs     = 500
lr         = 1e-4
momentum   = 0.9
w_decay    = 1e-5
step_size  = 9
gamma      = 0.5

net = fcn()
plus_epo = 0
sys_argv_len = len(sys.argv)
if sys_argv_len>4:
    if sys.argv[4] == 'load':
        print("_________________________________________________LOAD________________________________________________")
        plus_epo = int(sys.argv[5])
        for i in range(plus_epo//step_size):
            lr*=gamma
        net.load_state_dict(torch.load('./models/'+dataset+'_z'+str(z)+'_k'+str(k)+'_epo'+str(plus_epo)+'sigmoid.pt'))
        val2(net, val_label, val_x, val_xtc)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 9 epochs

early_stopping = 0
early_stopping2 = 0
last_val = 0
last_val2 = 0
print("START:", datetime.datetime.now())
for epoch in range(epochs):  # loop over the dataset multiple times
    #if early_stopping>=66:
    #    break
    running_loss = 0.0
    for i in range(training_size):
        # get the inputs
        inputX = Variable(train_x[i])
        #inputXTC = Variable(train_xtc[i])
        labelX = Variable(train_label[i])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #outputs = net(inputX, inputXTC)
        outputs = net(inputX)
        loss = criterion(outputs, labelX)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if (i+1) % 1000 ==0 :    # print every 900step
            print('[%d, %5d] loss: %.3f' %
                  (epoch +plus_epo+ 1, i + 1, running_loss/1000))
            running_loss = 0.0
    print("_________________________________")
    print(datetime.datetime.now())
    #v = val()
    v2 = val2(net, val_label, val_x, val_xtc)
    torch.save(net.state_dict(), './models/'+dataset+'_z'+str(z)+'_k'+str(k)+'_epo'+str(epoch+plus_epo+1)+'sigmoid.pt')
    """if(v>last_val):
        early_stopping = 0
        last_val = v
    else:
        early_stopping += 1"""

    if(v2>last_val2):
        early_stopping2 = 0
        last_val2 = v2
    else:
        early_stopping2 += 1
    if early_stopping2==0:
        torch.save(net.state_dict(), './models/best/'+dataset+'_z'+str(z)+'_k'+str(k)+'_sigmoid_best.pt')
    print('epo:', epoch+plus_epo+1, 'early stop:', early_stopping2)
    print("_________________________________")

print('Finished Training')
