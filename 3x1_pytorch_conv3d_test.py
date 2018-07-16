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

np.set_printoptions(threshold=256*256)
z = 1
label = []

temp_original = []
original = []

z_half = int(z/2)
print("Z:", z)
print(z_half)

def getData(fileName):
    xData = nib.load(fileName).get_data()
    temp = np.zeros((181+2*z_half,256,256))
    xData = xData - np.min(xData)
    temp[z_half:temp.shape[0]-z_half, ...] = xData
    #xData = temp

    for i in range(temp.shape[0]):
        original.append(temp[i, ...])
        

def getDataIBSR(fileName):
    xData = nib.load(fileName).get_data().swapaxes(0, 1) # test ibsr
    #xData = nib.load(fileName).get_data()
    temp = np.zeros((xData.shape[0]+2*z_half,xData.shape[1],256))
    xData = xData - np.min(xData)
    temp[z_half:temp.shape[0]-z_half, ...] = xData
    #xData = temp

    for i in range(temp.shape[0]):
        original.append(temp[i, ...])

label_val = 255

def getLabel(fileName):
    label_ = nib.load(fileName).get_data()
    label_ = np.where(((label_==1)|(label_==2)|(label_==3)), 1, 0)
    temp = np.zeros((362,512,512))
    temp[..., 40:512-38, 76:512-74] = label_
    label_ = temp
    newL = np.zeros((181+2*z_half,256,256))
    for i in range(z_half, int(label_.shape[0]/2)-z_half):
        newL[i,...] = np.where(scipy.misc.imresize(label_[2*(i-z_half),:,:], 0.5, interp='nearest')!=0, label_val, 0)
    
    for i in range(newL.shape[0]):
        label.append(newL[i, ...].reshape(256, 256, 1))
    
def getLabelIBSR(fileName):
    label_ = np.where(nib.load(fileName).get_data().swapaxes(0, 1)!=0 , label_val, 0) # test ibsr
    
    for i in range(label_.shape[0]):
        label.append(label_[i, ...].reshape(label_.shape[1], 256, 1))
"""'
minc20Files = sorted([f for f in listdir('../minc/20/') if isfile(join('../minc/20/', f))])
#print(minc20Files)
minc20Files = minc20Files[1:]
#minc20Files = minc20Files[:4]
print(minc20Files)
for i, mFile in enumerate(minc20Files):
    if mFile.find('crisp')!=-1:
        getLabel('../minc/20/'+mFile)
    elif mFile.find('t1w')!=-1:
        getData('../minc/20/'+mFile)'"""

getDataIBSR('../ibsr/10/IBSR_01/images/MINC/IBSR_01_ana.hdr.mnc')
getLabelIBSR('../ibsr/10/IBSR_01/segmentation/MINC/IBSR_01_segTRI_ana.hdr.mnc')

label = np.array(label)
original = np.array(original)


print(len(original), original[0].shape, label.shape)


train_x = []
train_label = []

val_x = []
val_label = []

def getTorchX():
    j = 0
    for i in range(z_half, original.shape[0]-z_half):
        if(j%10==0):
            val_x.append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,256,256))).float())
        else:
            train_x.append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,256,256))).float())
        j+=1

def getTorchLabel():
    j = 0
    for i in range(z_half, label.shape[0]-z_half):
        if(j%10==0):
            val_label.append(label[i, ...])
        else:
            train_label.append(torch.from_numpy(np.reshape(label[i, ...], (1, 1, 1,256, 256))).float())
        j+=1

def getTorchX2():
    for i in range(original.shape[0]):
        mi = i%(256+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(256+2*z_half)):
            continue
        else:
            val_x.append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,original.shape[1],256))).float())
            
def getTorchLabel2():
    for i in range(original.shape[0]):
        mi = i%(original.shape[0]+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(original.shape[0]+2*z_half)):
            continue
        else:
            val_label.append(label[i, ...])
            
def getTorchLabelIBSR():
    for i in range(label.shape[0]):
        val_label.append(label[i, ...])
getTorchX2()
getTorchLabelIBSR()
training_size = len(train_label)
print(len(val_label), val_label[0].shape)
print(len(val_x), val_x[0].shape)

class fcn(nn.Module):
    def __init__(self):
        super(fcn, self).__init__()

        # ------------------------------for main X------------------------------
        #self.max_pool = nn.MaxPool3d(kernel_size=(3,1,1), stride=(1,1,1))
        self.c0 = nn.Conv3d(1, 1, kernel_size=(z,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_0     = nn.BatchNorm3d(1)
        self.conv_1 = nn.Conv3d(1, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_1     = nn.BatchNorm3d(32)
        
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


    def forward(self, x):

        x = F.dropout3d(self.bn_0(F.relu(self.c0(x))), p=0.1)
        x = F.dropout3d(self.bn_1(F.relu(self.conv_1(x))), p=0.2)
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
        s3 = F.dropout3d(self.bn_8((self.upsample_4(s3))), p=0.2)
        return s3
        
def val():
    total = 0
    intersect = 0

    jtotal = 0
    jintersect = 0

    for i in range(len(val_label)):
        val_input = Variable(val_x[i])
        out = net(val_input).data.numpy()
        out = np.reshape(out,(val_x[i].numpy().shape[3],256))
        try:
            thresh = threshold_otsu(out)
        except:
            thresh = 255
        #print(thresh)
        #plt.imsave(arr=val_x[i].numpy().reshape(z,-1,256)[z_half,...], cmap='gray', fname='./test/'+str(i)+'_'+'input.jpg')
        #plt.imsave(arr=out, cmap='gray', fname='./test/'+str(i)+'_'+'b4.jpg')
        out = np.reshape(np.where(out>=thresh, 1, 0), (128,256))
        #print(out)
        #plt.imsave(arr=out, cmap='gray', fname='./test/'+str(i)+'_'+'.jpg')
        lab = np.reshape(np.where(val_label[i]==label_val, 1, 0), (128,256))
        #plt.imsave(arr=np.reshape(lab,(128,256)), cmap='gray', fname='./test/'+str(i)+'_'+'lab.jpg')
        total += np.sum(out) + np.sum(lab)
        temp = out+lab
        #print(temp)
        intersect += np.sum(np.where(temp==2, 1, 0))
        
        jtotal += np.sum(out) + np.sum(lab) - np.sum(np.where(temp==2, 1, 0))
        jintersect += np.sum(np.where(temp==2, 1, 0))

    return 2*intersect/total, jintersect/jtotal

net = fcn()
test_epo = 35
print("_________________________________________________TEST________________________________________________")
net.load_state_dict(torch.load('./models/z'+str(z)+'_epo'+str(test_epo)+'.pt'))
print('z'+str(z)+'_epo'+str(test_epo)+'.pt')
print(val())

