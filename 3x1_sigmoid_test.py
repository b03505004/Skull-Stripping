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



def getData(fileName, z_half, original):
    xData = nib.load(fileName).get_data()
    temp = np.zeros((181+2*z_half,256,256))
    xData = xData - np.min(xData)
    temp[z_half:temp.shape[0]-z_half, ...] = xData
    #xData = temp

    for i in range(temp.shape[0]):
        original.append(temp[i, ...])
        

label_val = 1

def getLabel(fileName, z, z_half, label):
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
    
def getDataIBSR(fileName, z, z_half, original):
    xData = nib.load(fileName).get_data().swapaxes(0, 1) # test ibsr
    #xData = nib.load(fileName).get_data()
    temp = np.zeros((xData.shape[0]+2*z_half,xData.shape[1],256))
    xData = xData - np.min(xData)
    temp[z_half:temp.shape[0]-z_half, ...] = xData
    #xData = temp

    for i in range(temp.shape[0]):
        original.append(temp[i, ...])

def getLabelIBSR(fileName, z, z_half, label):
    label_ = np.where(nib.load(fileName).get_data().swapaxes(0, 1)!=0 , label_val, 0) # test ibsr
    
    for i in range(label_.shape[0]):
        label.append(label_[i, ...].reshape(label_.shape[1], 256, 1))

shape0_20 = 64
def getDataIBSR20(fileName, z, z_half, original):
    xData = loadmat(fileName)['volume']
    temp = np.zeros((256, 256, xData.shape[2]+2*z_half))
    xData = xData - np.min(xData)
    temp[..., z_half:temp.shape[2]-z_half] = xData
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

"""for _ in zs:
    xs.append([])
    labels.append([])"""

num_pic = 55
def getTorchX(z, z_half, original, val_x):
    val_x.append([])
    for i in range(original.shape[0]):
        #mi = i%(num_pic+2*z_half)
        #if ((mi-z_half)<0) or ((mi+z_half)>=(num_pic+2*z_half)):
        mi = i%(original.shape[0])
        if ((mi-z_half)<0) or ((mi+z_half)>=(original.shape[0])):
            continue
        else:
            #print(i)
            #print(original.shape)
            val_x[-1].append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,original.shape[1],256))).float())

            
def getTorchLabelIBSR(z, z_half, label, val_label):
    val_label.append([])
    for i in range(label.shape[0]):
        val_label[-1].append(label[i, ...])


class fcn(nn.Module):
    def __init__(self, z):
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


def val_new(net, val_label, val_x, z, z_half):
    for b,brain in enumerate(val_label):
        tP = 0.0
        tN = 0.0
        fP = 0.0
        fN = 0.0

        for i in range(len(brain)):
            val_input = Variable(val_x[b][i])
            out = net(val_input).data.numpy()
            out = np.reshape(out,(val_x[b][i].numpy().shape[3],256))
            try:
                thresh = threshold_otsu(out)
            except:
                thresh = 255
            #plt.imsave(arr=val_x[b][i].numpy().reshape(z,-1,256)[z_half,...], cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'input.jpg')
            #plt.imsave(arr=out, cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'b4.jpg')
            out = np.reshape(np.where(out>=thresh, 13, 7), (-1,256))
            #plt.imsave(arr=out, cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'.jpg')
            lab = np.reshape(np.where(brain[i]==label_val, 3, 1), (-1,256))
            #plt.imsave(arr=np.reshape(lab,(-1,256)), cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'lab.jpg')
            temp = out+lab
            #print(temp)
            tP += np.sum(np.where(temp==16, 1, 0))
            tN += np.sum(np.where(temp==8, 1, 0))
            fP += np.sum(np.where(temp==14, 1, 0))
            fN += np.sum(np.where(temp==10, 1, 0))
    dice = 2*tP/(2*tP+fP+fN)
    jaccard = tP/(tP+fP+fN)
    sensitivity = tP/(tP+fN)
    specificity = tN/(tN+fP)
    conformity = 1-((fP+fN)/tP)
    sensibility = 1-(fP/(tP+fN))
    print("dice:",dice,"jaccard:",jaccard,"sensitivity:",sensitivity,"specificity:",\
    specificity,"conformity:",conformity,"sensibility:", sensibility)

def val_ensemble(net, val_label, val_x, zl):
    for b,brain in enumerate(val_label):
        tP = 0.0
        tN = 0.0
        fP = 0.0
        fN = 0.0
        
        for i in range(len(brain)):
            out = np.zeros((val_label[b][i].shape[0],256))
            for z in range(zl):
                #print(z)
                val_input = Variable(val_x[z][b][i])
                outtemp = net[z](val_input).data.numpy()
                outtemp = np.reshape(outtemp,(out.shape[0],256))
                out += outtemp
            
            plt.imsave(arr=val_x[0][b][i].numpy().reshape(z,-1,256)[0,...], cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'input.jpg')
            plt.imsave(arr=out, cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'b4.jpg')
            out = np.reshape(np.where(out>=0.5*zl, 13, 7), (-1,256))
            plt.imsave(arr=out, cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'.jpg')
            lab = np.reshape(np.where(brain[i]==label_val, 3, 1), (-1,256))
            plt.imsave(arr=np.reshape(lab,(-1,256)), cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'lab.jpg')
            temp = out+lab
            #print(temp)
            tP += np.sum(np.where(temp==16, 1, 0))
            tN += np.sum(np.where(temp==8, 1, 0))
            fP += np.sum(np.where(temp==14, 1, 0))
            fN += np.sum(np.where(temp==10, 1, 0))
    dice = 2*tP/(2*tP+fP+fN)
    jaccard = tP/(tP+fP+fN)
    sensitivity = tP/(tP+fN)
    specificity = tN/(tN+fP)
    conformity = 1-((fP+fN)/tP)
    sensibility = 1-(fP/(tP+fN))
    print("dice:",dice,"jaccard:",jaccard,"sensitivity:",sensitivity,"specificity:",\
    specificity,"conformity:",conformity,"sensibility:", sensibility)

ks = [0]
if dataset=="ibsr20":
    KK = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
elif dataset=='lpba':
    KK = [str(i)+str(j) for i in range(4) for j in range(10)][1:]
    KK.append("40")
half_lk = int(len(KK)//2)
print(half_lk)
if ks[0] == 0:
    KK = KK[half_lk:]
else:
    KK = KK[:half_lk]

xs = []
labels = []
nets = []
for i,z in enumerate(zs):
    xs.append([])
    #labels.append([])

    print("__________________________________________")
    print("Z:", zs[i])
    for k in ks:
        print("K:", k)
        #for brain_num in ['01', '02', '03', '04']:
        
        if dataset=="ibsr20":
            for brain_num in KK:
                #print(brain_num)
                temp_original = []
                getDataIBSR20('../ibsr/20mat/IBSR'+brain_num+'_1', zs[i], z_halfs[i], temp_original)
                temp_original = np.array(temp_original)
                #print(temp_original.shape, temp_lab.shape)
                getTorchX(zs[i], z_halfs[i], temp_original, xs[-1])
                if i == 0:
                    temp_lab = []
                    getLabelIBSR20('../ibsr/20mat/IBSR'+brain_num+'_gt', zs[i], z_halfs[i], temp_lab)
                    temp_lab = np.array(temp_lab)
                    getTorchLabelIBSR(zs[i], z_halfs[i], temp_lab, labels)
        elif dataset=='lpba':
            for brain_num in KK:
                #print(brain_num)
                temp_original = []
                getDataLPBA40('../lpba/lpba_img/S'+brain_num+'.native.mri.img', zs[i], z_halfs[i], temp_original)
                temp_original = np.array(temp_original)
                #print(temp_original.shape, temp_lab.shape)
                getTorchX(zs[i], z_halfs[i], temp_original, xs[-1])
                if i == 0:
                    temp_lab = []
                    getLabelLPBA40('../lpba/lpba_img/S'+brain_num+'.native.tissue.img', zs[i], z_halfs[i], temp_lab)
                    temp_lab = np.array(temp_lab)
                    getTorchLabelIBSR(zs[i], z_halfs[i], temp_lab, labels)


        temp_net = fcn(z)
        temp_net.load_state_dict(torch.load('./models/best/'+dataset+'_z'+str(z)+'_k'+str(k)+'_sigmoid_best.pt'))
        nets.append(temp_net)
        #val_new(temp_net, labels[i], xs[i], z, z_halfs[i])
for n in nets:
    print(n)
val_ensemble(nets, labels, xs, len(zs))
