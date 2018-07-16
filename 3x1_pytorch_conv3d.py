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

np.set_printoptions(threshold=256*256)
z = 5
label = []
original = []

labelIBSR = []
originalIBSR = []

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


def getDataIBSR(fileName):
    xData = nib.load(fileName).get_data().swapaxes(0, 1) # test ibsr
    #xData = nib.load(fileName).get_data()
    temp = np.zeros((xData.shape[0]+2*z_half,xData.shape[1],256))
    xData = xData - np.min(xData)
    temp[z_half:temp.shape[0]-z_half, ...] = xData
    #xData = temp

    for i in range(temp.shape[0]):
        originalIBSR.append(temp[i, ...])

def getLabelIBSR(fileName):
    label_ = np.where(nib.load(fileName).get_data().swapaxes(0, 1)!=0 , label_val, 0) # test ibsr
    temp = np.zeros((256+2*z_half,128,256))
    temp[z_half:temp.shape[0]-z_half, ...] = label_

    for i in range(temp.shape[0]):
        labelIBSR.append(temp[i, ...].reshape(temp.shape[1], 256, 1))

minc20Files = sorted([f for f in listdir('../minc/20/') if isfile(join('../minc/20/', f))])
#print(minc20Files)
minc20Files = minc20Files[1:]
#minc20Files = minc20Files[:4]
print(minc20Files)
for i, mFile in enumerate(minc20Files):
    if mFile.find('crisp')!=-1:
        getLabel('../minc/20/'+mFile)
    elif mFile.find('t1w')!=-1:
        getData('../minc/20/'+mFile)


for i in range(1, 10):
    filePathX = '../ibsr/10/IBSR_0'+str(i)+'/images/MINC/IBSR_0'+str(i)+'_ana.hdr.mnc'
    filePathSeg = '../ibsr/10/IBSR_0'+str(i)+'/segmentation/MINC/IBSR_0'+str(i)+'_segTRI_ana.hdr.mnc'
    getDataIBSR(filePathX)
    getLabelIBSR(filePathSeg)

label = np.array(label)
original = np.array(original)
labelIBSR = np.array(labelIBSR)
originalIBSR = np.array(originalIBSR)

print("BrainWeb:", original.shape, label.shape)
print("IBSR:", originalIBSR.shape, labelIBSR.shape)

train_x = []
train_label = []

val_x = []
val_label = []

def getTorchX():
    for i in range(original.shape[0]):
        mi = i%(181+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(181+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_x.append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,original.shape[1],256))).float())
                #plt.imsave(arr=original[i, ...], cmap='gray', fname='./all/'+str(i)+'_x.jpg')
            else:
                train_x.append(torch.from_numpy(np.reshape(original[i-z_half : i+z_half+1,...], (1,1,z,original.shape[1],256))).float())
                #plt.imsave(arr=original[i, ...], cmap='gray', fname='./all/'+str(i)+'_x.jpg')

def getTorchLabel():
    for i in range(original.shape[0]):
        mi = i%(181+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(181+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_label.append(label[i, ...])
                #plt.imsave(arr=label[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(i)+'_lab.jpg')
            else:
                train_label.append(torch.from_numpy(np.reshape(label[i, ...], (1, 1, 1, label.shape[1], 256))).float())
                #plt.imsave(arr=label[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(i)+'_lab.jpg')

def getTorchXIBSR():
    for i in range(originalIBSR.shape[0]):
        mi = i%(256+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(256+2*z_half)):
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
        mi = i%(256+2*z_half)
        if ((mi-z_half)<0) or ((mi+z_half)>=(256+2*z_half)):
            continue
        else:
            if (i%10==0):
                val_label.append(labelIBSR[i, ...])
                #plt.imsave(arr=labelIBSR[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(181*20+i)+'_lab.jpg')
            else:
                train_label.append(torch.from_numpy(np.reshape(labelIBSR[i, ...], (1, 1, 1, labelIBSR.shape[1], 256))).float())
                #plt.imsave(arr=labelIBSR[i, ...].reshape(-1,256), cmap='gray', fname='./all/'+str(181*20+i)+'_lab.jpg')

def shuffle(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a,b

getTorchX()
getTorchLabel()
print('Train label:', len(train_label), train_label[0].shape, len(val_label))
print('Train x:', len(train_x), train_x[0].shape, len(val_x))
print('Val label:', len(val_label), val_label[0].shape)
print('Val x:', len(val_x), val_x[0].shape)
getTorchLabelIBSR()
getTorchXIBSR()

shuffle(train_label, train_x)
training_size = len(train_label)

print('Train label:', len(train_label), train_label[0].shape, len(val_label))
print('Train x:', len(train_x), train_x[0].shape, len(val_x))
print('Val label:', len(val_label), val_label[0].shape)
print('Val x:', len(val_x), val_x[0].shape)

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
    return 2*union/total


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
if sys_argv_len>1:
    if sys.argv[1] == 'load':
        print("_________________________________________________LOAD________________________________________________")
        plus_epo = 24
        net.load_state_dict(torch.load('./models/z'+str(z)+'_epo'+str(plus_epo)+'.pt'))
        print(val())

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 9 epochs

early_stopping = 0
last_val = 0
for epoch in range(epochs):  # loop over the dataset multiple times
    #if early_stopping>=66:
    #    break
    running_loss = 0.0
    for i in range(training_size):
        # get the inputs
        inputX = Variable(train_x[i])
        labelX = Variable(train_label[i])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputX)
        loss = criterion(outputs, labelX)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if (i+1) % 1000 ==0 :    # print every 900step
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss/1000))
            running_loss = 0.0
    v = val()
    torch.save(net.state_dict(), './models/z'+str(z)+'_epo'+str(epoch+plus_epo+1)+'.pt')
    if(v>last_val):
        early_stopping = 0
        last_val = v
    else:
        early_stopping += 1
    print('epo:', epoch+plus_epo+1, 'val dice coefficient:', v, 'early stop:', early_stopping)

print('Finished Training')
