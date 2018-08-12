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
zs = [1,3,5]
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
        

label_val = 255

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
    temp = label_
    #temp[:, :, :] = label_

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
        self.c0 = nn.Conv3d(1, 16, kernel_size=(z,1,1), stride=(1,1,1))
        self.bn_0     = nn.BatchNorm3d(16)
        self.c02 = nn.Conv3d(16, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.bn_02     = nn.BatchNorm3d(32)
        self.conv_1 = nn.Conv3d(32, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
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
        x = F.dropout3d(self.bn_02(F.relu(self.c02(x))), p=0.2)
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
        
shape0 = 256
def val(net, val_label, val_x, z, z_half):
    dicecoef_sum = 0
    jaccard_sum = 0
    for b,brain in enumerate(val_label):
        total = 0
        intersect = 0

        jtotal = 0
        jintersect = 0
        for i in range(len(brain)):
            val_input = Variable(val_x[b][i])
            out = net(val_input).data.numpy()
            out = np.reshape(out,(val_x[b][i].numpy().shape[3],256))
            try:
                thresh = threshold_otsu(out)
            except:
                thresh = 255
            #print(thresh)
            #plt.imsave(arr=val_x[b][i].numpy().reshape(z,-1,256)[z_half,...], cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'input.jpg')
            #plt.imsave(arr=out, cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'b4.jpg')
            out = np.reshape(np.where(out>=thresh, 1, 0), (-1,256))
            #print(out)
            #plt.imsave(arr=out, cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'.jpg')
            lab = np.reshape(np.where(brain[i]==label_val, 1, 0), (-1,256))
            #plt.imsave(arr=np.reshape(lab,(-1,256)), cmap='gray', fname='./test/'+str(b)+'_'+str(i)+'_'+'lab.jpg')
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

def val_new(net, val_label, val_x, z, z_half):
    dice = 0.0
    jaccard = 0.0
    sensitivity = 0.0
    specificity = 0.0
    conformity = 0.0
    sensibility = 0.0
    for b,brain in enumerate(val_label):
        #print(len(brain))
        #print(len(val_x[b]))
        #print(b)
        tP = 0.0
        tN = 0.0
        fP = 0.0
        fN = 0.0

        for i in range(len(brain)):
            #print("i", i)
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
        print(b, 2*tP/(2*tP+fP+fN))
        dice += 2*tP/(2*tP+fP+fN)
        jaccard += tP/(tP+fP+fN)
        sensitivity += tP/(tP+fN)
        specificity += tN/(tN+fP)
        conformity += 1-((fP+fN)/tP)
        sensibility += 1-(fP/(tP+fN))
    numOfBrain = len(val_label)
    dice = dice/numOfBrain
    jaccard = jaccard/numOfBrain
    sensitivity = sensitivity/numOfBrain
    specificity = specificity/numOfBrain
    conformity = conformity/numOfBrain
    sensibility = sensibility/numOfBrain
    print("dice:",dice,"jaccard:",jaccard,"sensitivity:",\
    sensitivity,"specificity:", specificity,\
    "conformity:",conformity,"sensibility:", sensibility)
    return dice,jaccard,sensitivity,specificity,conformity,sensibility


ks = [0, 1, 2, 3, 4]
if dataset=="ibsr20":
    KK = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']
elif dataset=='lpba':
    KK = [str(i)+str(j) for i in range(4) for j in range(10)][1:]
    KK.append("40")

for i,z in enumerate(zs):
    print("__________________________________________")
    print("Z:", zs[i])

    dice = 0.0
    jaccard = 0.0
    sensitivity = 0.0
    specificity = 0.0
    conformity = 0.0
    sensibility = 0.0

    for k in ks:
        xs = []
        labels = []
        print(len(xs))
        print("K:", k)
        #for brain_num in ['01', '02', '03', '04']:
        
        if dataset=="ibsr20":
            for brain_num in KK[k*4:k*4+4]:
                #print(brain_num)
                temp_original = []
                temp_lab = []
                getDataIBSR20('../ibsr/20mat/IBSR'+brain_num+'_1', zs[i], z_halfs[i], temp_original)
                getLabelIBSR20('../ibsr/20mat/IBSR'+brain_num+'_gt', zs[i], z_halfs[i], temp_lab)
                temp_original = np.array(temp_original)
                temp_lab = np.array(temp_lab)
                #print(temp_original.shape, temp_lab.shape)
                getTorchX(zs[i], z_halfs[i], temp_original, xs)
                getTorchLabelIBSR(zs[i], z_halfs[i], temp_lab, labels)
        elif dataset=='lpba':
            for brain_num in KK[k*8:k*8+8]:
                #print(brain_num)
                temp_original = []
                temp_lab = []
                getDataLPBA40('../lpba/lpba_img/S'+brain_num+'.native.mri.img', zs[i], z_halfs[i], temp_original)
                getLabelLPBA40('../lpba/lpba_img/S'+brain_num+'.native.tissue.img', zs[i], z_halfs[i], temp_lab)
                temp_original = np.array(temp_original)
                temp_lab = np.array(temp_lab)
                #print(temp_original.shape, temp_lab.shape)
                getTorchX(zs[i], z_halfs[i], temp_original, xs)
                getTorchLabelIBSR(zs[i], z_halfs[i], temp_lab, labels)
                #print(len(xs))

        #print(len(xs[0]), len(labels[0]))
        #print("______________________________________")
        nets = []
        #epos = [20, 20, 14, 8, 15]
        #epos = [12]
        temp_net = fcn(z)
        #for normal test
        #temp_net.load_state_dict(torch.load('./models/z'+str(z)+'_k'+str(k)+'_epo'+str(epos[i])+'.pt'))
        #for best test
        temp_net.load_state_dict(torch.load('./models/best/'+dataset+'_z'+str(z)+'_k'+str(k)+'.pt'))
        #nets.append(temp_net)
        #val(nets[i], labels[i], xs[i], z, z_halfs[i])
        d,j,s1,s2,c,s3 = val_new(temp_net, labels, xs, z, z_halfs[i])
        #print(d)
        dice += d
        jaccard += j
        sensitivity += s1
        specificity += s2
        conformity += c
        sensibility += s3
    nk = len(ks)
    print("AVG: dice:",dice/nk,"jaccard:",jaccard/nk,"sensitivity:",\
    sensitivity/nk,"specificity:", specificity/nk,\
    "conformity:",conformity/nk,"sensibility:", sensibility/nk)
    print("_________________________________________________________")
    
"""
def val_ensemble():
    total = 0
    intersect = 0
    jtotal = 0
    jintersect = 0

    for i in range(len(val_label1)):
        val_input1 = Variable(val_x1[i])
        val_input3 = Variable(val_x3[i])
        val_input5 = Variable(val_x5[i])

        out1 = net1(val_input1).data.numpy()
        out1 = np.reshape(out1,(val_x1[i].numpy().shape[3],256))
        out3 = net3(val_input3).data.numpy()
        out3 = np.reshape(out3,(val_x3[i].numpy().shape[3],256))
        out5 = net5(val_input5).data.numpy()
        out5 = np.reshape(out5,(val_x5[i].numpy().shape[3],256))
        out = out1 + out3 + out5
        try:
            thresh = threshold_otsu(out)
        except:
            thresh = 255
        try:
            thresh1 = threshold_otsu(out1)
        except:
            thresh1 = 255
        try:
            thresh3 = threshold_otsu(out3)
        except:
            thresh3 = 255
        try:
            thresh5 = threshold_otsu(out5)
        except:
            thresh5 = 255
        #print(thresh)
        plt.imsave(arr=val_x1[i].numpy().reshape(z1,-1,256)[z_half1,...], cmap='gray', fname='./plot_mat20/ens'+str(i)+'_'+'input.jpg')
        plt.imsave(arr=out, cmap='gray', fname='./plot_mat20/ens'+str(i)+'_'+'b4.jpg')
        out1 = np.reshape(np.where(out1>=thresh1, 1, 0), (shape0,256))
        out3 = np.reshape(np.where(out3>=thresh3, 1, 0), (shape0,256))
        out5 = np.reshape(np.where(out5>=thresh5, 1, 0), (shape0,256))
        t = out1+out3+out5
        out2 = np.reshape(np.where(t>=2, 1, 0), (shape0,256))
        plt.imsave(arr=out2, cmap='gray', fname='./plot_mat20/ens'+str(i)+'_'+'2.jpg')
        out = np.reshape(np.where(out>=thresh, 1, 0), (shape0,256))
        #print(out)
        plt.imsave(arr=out, cmap='gray', fname='./plot_mat20/ens'+str(i)+'_'+'.jpg')
        lab = np.reshape(np.where(val_label1[i]==label_val, 1, 0), (shape0,256))
        plt.imsave(arr=np.reshape(lab,(shape0,256)), cmap='gray', fname='./plot_mat20/ens'+str(i)+'_'+'lab.jpg')
        total += np.sum(out2) + np.sum(lab)
        temp = out2+lab
        #print(temp)
        intersect += np.sum(np.where(temp==2, 1, 0))
        
        jtotal += np.sum(out2) + np.sum(lab) - np.sum(np.where(temp==2, 1, 0))
        jintersect += np.sum(np.where(temp==2, 1, 0))
    return 2*intersect/total, jintersect/jtotal
#print(val_ensemble())
"""