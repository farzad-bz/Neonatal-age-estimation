import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from medpy.metric.binary import dc,hd,asd,assd
import nibabel as nib
import SimpleITK as sitk

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 2x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceF = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceB[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceF[i, 0] = self.inter(pred[i, 1], GT[i, 1])

            DiceB[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceF[i, 1] = self.sum(pred[i, 1], GT[i, 1])

        return DiceB, DiceF


class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1 
        # pred is converted to 0 and 1
        # Note: works for 3D dataset as well 4x152x200x200
        batchsize = GT.size(0)
        Dices = []
        for i in range(4):
            Dices.append(to_var(torch.zeros(batchsize, 2)))
 

        for i in range(batchsize):
            for j in range(4):
                Dices[j][i, 0] = self.inter(pred[i, j], GT[i, j])

                Dices[j][i, 1] = self.sum(pred[i, j], GT[i, j])

        return tuple(Dices)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


class Hausdorff(nn.Module):
    def __init__(self, dist=320 ** 2):
        super(Hausdorff, self).__init__()
        self.dist = dist

    def edgePixels(self, img, channel):  # , GT):
        # return the list of pixels from the edge of the surface
        padVal = 0
        # changing padValue for channel 0 to avoid detecting edge of the image
        if channel == 0:
            padVal = 1
        sub = F.pad(img[1:, :], (0, 0, 0, 1), value=padVal) + F.pad(img[:-1, :], (0, 0, 1, 0), value=padVal) + \
              F.pad(img[:, 1:], (0, 1, 0, 0), value=padVal) + F.pad(img[:, :-1], (1, 0, 0, 0), value=padVal)
        edge = 4 * img - sub.data

        return edge.clamp(0, 1).nonzero().float()

    def maxminDist(self, edge_pred, edge_GT):
        dists = (edge_GT.expand(len(edge_pred), *edge_GT.size()) - edge_pred.unsqueeze(1)).pow(2).sum(-1)

        return dists.min(-1)[0].max()

    def forward(self, pred, GT):
        # Computes the Hausdorff distance between two segmentations
        # The inputs are Variables with N channels corresponding to N classes
        # for each pixel, only one channel must be equal to 1 and all others to 0
        batchsize, n_classes, _, _ = pred.size()
        hausdorffs = to_var(torch.zeros(batchsize, n_classes))

        for i in range(batchsize):
            for j in range(n_classes):
                edge_pred = self.edgePixels(pred.data[i, j], j)
                if len(edge_pred) == 0:
                    continue
                edge_GT = self.edgePixels(GT.data[i, j], j)
                if len(edge_GT) == 0:
                    continue

                hausdorffs[i, j] = self.maxminDist(edge_pred, edge_GT)

        return hausdorffs.sum() / self.dist


def getSingleInputImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(3))
    Val[1] = 1.0

    x = pred

    out = x * Val.view(1, 3, 1, 1)
    return out.sum(dim=1, keepdim=True)


def computeDSC(pred, gt):

    dscAll= np.zeros((pred.shape[0],4))

    for i_b in range(pred.shape[0]):
        gt_id = (gt[i_b, 0, :]/0.005).round()
        for i_c in range(pred.shape[1]-1):
            pred_id = pred[i_b,i_c+1,:]
            gt_class = np.zeros((gt_id.cpu().data.numpy().shape))
            idx = np.where(gt_id.cpu().data.numpy()==(i_c+1))
            gt_class[idx]=1
            dscAll[i_b,i_c]=(dc(pred_id.cpu().data.numpy(),gt_class))

    return dscAll.mean(axis=0)



def getSingleImageBin(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0
    
    x = predToSegmentation(pred)

    out = x * Val.view(1, 2, 1, 1)
    return out.sum(dim=1, keepdim=True)


def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    

    # Heart
    # Val[1] = 0.33333334
    # Val[2] = 0.66666669
    # Val[3] = 1.0

    # Bladder
    #Val[1] = 0.3137255
    #Val[2] = 0.627451
    #Val[3] = 0.94117647

    # BraTS
    Val = torch.from_numpy(np.arange(10, dtype=np.float32))

    x = predToSegmentation(pred)

    out = x * Val.view(1, 10, 1, 1, 1)
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotTumorClass(batch):
    data = batch.cpu().data.numpy()
    classLabels = np.zeros((data.shape[0], 2))

    tumorVal = 1.0
    for i in range(data.shape[0]):
        img = data[i, :, :, :]
        values = np.unique(img)
        if len(values) > 3:
            classLabels[i, 1] = 1
        else:
            classLabels[i, 0] = 1

    tensorClass = torch.from_numpy(classLabels).float()

    return Variable(tensorClass.cuda())


def getOneHotSegmentation(batch, num_classes):
    backgroundVal = 0
    # Heart
    #label1 = 0.33333334
    #label2 = 0.66666669
    #label3 = 1.0

    # Bladder
    #label1 = 0.3137255
    #label2 = 0.627451
    #label3 = 0.94117647

    # BraTS
    labels = [i for i in range(num_classes)]

    #DHCP-9class
    #labels = [i for i in range(1,3)]


    batch = batch.unsqueeze(dim=1)
    oneHotLabels = torch.cat(tuple([batch == i for i in labels]), dim=1)

    #oneHotLabels = torch.cat((batch < label1, batch >= label1), dim=1)
    #batch = batch.unsqueeze(dim=1)
    #oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3), dim=1)
    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    # Bladder
    # spineLabel = 0.33333334

    # BraTS, input (label) is already discrete values : 0, 1, 2 and 3
    #spineLabel = 1.0
    spineLabel = 1.0
    return (batch / spineLabel).round().long().squeeze()

from scipy import ndimage


def evaluateBraTS(imageDataCNN, imageDataGT):

    #pathGT = os.path.join('../brats_data',mode,'NIFTI')
    #pathCNN = os.path.join('./Results/Images/NIFTI',modelname,mode)
    #subjNames = os.listdir(pathGT)


    numClasses = 9
    DSC = np.zeros((len(imageDataCNN), numClasses))

    #subjNames.sort()
    labelsCNN = np.arange(10, dtype=np.float32)
    labelsGT = np.arange(10, dtype=np.float32)

    for s_i in range(len(imageDataCNN)):
         #path_Subj_GT = os.path.join(pathGT, subjNames[s_i])
         #path_Subj_CNN = os.path.join(pathCNN, subjNames[s_i])

         #[imageDataGT, img_proxy] = load_nii(path_Subj_GT, printFileNames=False)
         #[imageDataCNN, img_proxy] = load_nii(path_Subj_CNN, printFileNames=False)

         label_GT_WT = np.zeros(imageDataGT.shape, dtype=np.int8)
         label_CNN_WT = np.zeros(imageDataCNN.shape, dtype=np.int8)

         label_GT_TC = np.zeros(imageDataGT.shape, dtype=np.int8)
         label_CNN_TC = np.zeros(imageDataCNN.shape, dtype=np.int8)

         label_GT_ET = np.zeros(imageDataGT.shape, dtype=np.int8)
         label_CNN_ET = np.zeros(imageDataCNN.shape, dtype=np.int8)

         idx_GT1 = np.where(imageDataGT == labelsGT[1])
         idx_GT2 = np.where(imageDataGT == labelsGT[2])
         idx_GT3 = np.where(imageDataGT == labelsGT[3])

         idx_CNN1 = np.where(imageDataCNN == labelsCNN[1])
         idx_CNN2 = np.where(imageDataCNN == labelsCNN[2])
         idx_CNN3 = np.where(imageDataCNN == labelsCNN[3])

         # Whole Tumor (1,2,3)
         label_GT_WT[idx_GT1]=1
         label_GT_WT[idx_GT2]=1
         label_GT_WT[idx_GT3]=1

         label_CNN_WT[idx_CNN1] = 1
         label_CNN_WT[idx_CNN2] = 1
         label_CNN_WT[idx_CNN3] = 1

         # Core Tumor (2,3)
         label_GT_TC[idx_GT2] = 1
         label_GT_TC[idx_GT3] = 1

         label_CNN_TC[idx_CNN2] = 1
         label_CNN_TC[idx_CNN3] = 1

         # Enhanced Tumor (3)
         label_GT_ET[idx_GT3] = 1

         label_CNN_ET[idx_CNN3] = 1

         DSC[s_i, 0] = dc(label_GT_WT, label_CNN_WT)
         DSC[s_i, 1] = dc(label_GT_TC, label_CNN_TC)
         DSC[s_i, 2] = dc(label_GT_ET, label_CNN_ET)

    return DSC.mean(axis=0)



def saveSegImages(net, img_batch, batch_size, epoch, modelName, deepSupervision=0):
    # print(" Saving images.....")
    #path = 'Results/ENet-Original'
    path = './Results_eval/' + modelName + '_' + str(epoch)
    if not os.path.exists(path):
        os.makedirs(path)

    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    times = []
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)

        if not deepSupervision:
            # No deep supervision
            tic = time.clock()
            segmentation_prediction = net(MRI)
            toc = time.clock()
            times.append(toc-tic)
        else:
            # Deep supervision
            segmentation_prediction, seg_3, seg_2, seg_1 = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImage(pred_y)

        # segmentation = getSingleImageBin(segmentation_prediction)

        #out = torch.cat((getSingleInputImage(MRI), segmentation, Segmentation))
        out = segmentation

        torchvision.utils.save_image(out.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)
        torchvision.utils.save_image(MRI.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '_Img.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)
        torchvision.utils.save_image(Segmentation.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '_GT.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)

    printProgressBar(total, total, done="Images saved !")


def saveImages(net, img_batch, batch_size, epoch, modelName, deepSupervision=0):
    print(" Saving images.....")
    path = './Results/Images_PNG/' + modelName + '_' + str(epoch)
    if not os.path.exists(path):
        os.makedirs(path)

    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    times = []
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        #image, labels, img_names = data
        image = torch.stack(data[:4], dim=1)
        labels = data[4]
        img_names = data[5]

        MRI = to_var(image)
        Segmentation = to_var(labels)
            
        # No deep supervision
        tic = time.clock()
        segmentation_prediction = net(MRI)
        toc = time.clock()
        times.append(toc-tic)

        pred_y = softMax(segmentation_prediction)
        segmentation_pred = getSingleImage(pred_y)
        
        # segmentation = getSingleImageBin(segmentation_prediction)
        
        #out = torch.cat((getSingleInputImage(MRI), segmentation_pred, Segmentation))
        #out = torch.cat((MRI[:,0], MRI[:,1], MRI[:,2], MRI[:,3], segmentation_pred[:,0], Segmentation))
        # (B, 4, 152, 200, 200)
        r = np.random.randint(30,130)
        #out = torch.cat((MRI[:,0], MRI[:,1], MRI[:,2], MRI[:,3], segmentation_pred[:,0], Segmentation))
        for j in range(batch_size):
            #out = torch.cat((MRI[:,j:j+1], segmentation_pred, Segmentation.unsqueeze(dim=1)), dim=1)
            #out = torch.cat((MRI[:,0,r], MRI[:,1,r], MRI[:,2,r], MRI[:,3,r], segmentation_pred[:,0,r], Segmentation[:,r]), dim=0)
            out = torch.cat((MRI[j:j+1,0,r], MRI[j:j+1,1,r], MRI[j:j+1,2,r], MRI[j:j+1,3,r], segmentation_pred[j:j+1,0,r]/3.0, Segmentation[j:j+1,r]/3.0), dim=0)
            out = out.unsqueeze(dim=1)
            torchvision.utils.save_image(out.data, os.path.join(path, str(i*batch_size+j) + '_Ep_' + str(epoch) + '.png'),
                                     nrow=3,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)

    printProgressBar(total, total, done="Images saved !")


def inferenceBinary(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)
    print("total : ", total)
    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)

    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHotBinary().cuda()
    softMax = nn.Softmax().cuda()
    timesAll = []
    start_time = time.time()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        if deepSupervision == False:
            segmentation_prediction = net(MRI)
        else:
            segmentation_prediction, seg3, seg2, seg1 = net(MRI)

        pred_y = softMax(segmentation_prediction)

        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesF, DicesB = dice(pred_y, Segmentation_planes)

        Dice1[i] = DicesF.data
        Dice2[i] = DicesB.data

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    timesAll = time.time() - start_time
    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    #print(' Mean time per slice is: {} s'.format(timesAll / i))

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)

    return [ValDice1, ValDice2]

def inference(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()
    timesAll = []

    start_time = time.time()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        pred_y = softMax(segmentation_prediction)
        
        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesN, Dices1, Dices2, Dices3= dice(pred_y, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    timesAll = time.time()-start_time
    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    #print(' Mean time per slice is: {} s'.format(timesAll/i))
    
    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3]


def inferenceVolume(net, img_batch, batch_size, epoch):
    total = len(img_batch)

    DiceL1 = torch.zeros(total, 2)
    DiceL2 = torch.zeros(total, 2)
    DiceL3 = torch.zeros(total, 2)
    DSC_brats = np.zeros((total, 3))
    
    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    imagesAll = []
    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()

    start_time = time.time()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        #image, labels, img_names = data
        image = torch.stack(data[:4], dim=1)
        labels = data[4]
        img_names = data[5]

        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)
        
        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)

        predDiscrete = predToSegmentation(pred_y)
        Segmentation_planes = getOneHotSegmentation(Segmentation)
        segmentation_pred = getSingleImage(pred_y)
        
        #pdb.set_trace()

        # DicesN, Dices1, Dices2, Dices3 = dice(pred_y, Segmentation_planes)
        DicesN, Dices1, Dices2, Dices3 = dice(predDiscrete, Segmentation_planes)

        DiceL1[i] = Dices1.data
        DiceL2[i] = Dices2.data
        DiceL3[i] = Dices3.data
    
        DSC_brats[i] = evaluateBraTS(segmentation_pred[:,0].data.cpu().numpy(), Segmentation.data.cpu().numpy())
        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    timesAll = time.time()-start_time
    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    print(' Mean time per slice is: {} s'.format(timesAll/i))

    ValDice1 = DicesToDice(DiceL1)
    ValDice2 = DicesToDice(DiceL2)
    ValDice3 = DicesToDice(DiceL3)

    DSC_brats_avg = DSC_brats.mean(axis=0)
   
    return [ValDice1,ValDice2,ValDice3, DSC_brats_avg]

def _to_itk(x):
    x = sitk.GetImageFromArray(x)
    x.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    x.SetOrigin((0.0, 0.0, 0.0))
    x.SetSpacing((1.0, 1.0, 1.0))
    return x

def inferenceVolumeTest(net, img_batch, batch_size, epoch):
    total = len(img_batch)
    path = './Results/Nifti/UNet_' + str(epoch)
    DiceL1 = torch.zeros(total, 2)
    DiceL2 = torch.zeros(total, 2)
    DiceL3 = torch.zeros(total, 2)

    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    imagesAll = []
    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()

    numClasses = 3
    DSC = np.zeros((len(img_batch), numClasses))

    start_time = time.time()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        # image, labels, img_names = data
        image = torch.stack(data[:4], dim=1)
        labels = data[4]
        img_names = data[5]

        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)   # (B, l, w, h, d)

        # predDiscrete = predToSegmentation(pred_y)
        #Segmentation_planes = getOneHotSegmentation(Segmentation)
        #segmentation_pred = getSingleImage(pred_y)
        pred_y = torch.argmax(pred_y, dim=1)   # (B, w, h, d)
        
        '''
        xform = np.eye(4) * 1
        imgNifti = nib.Nifti1Image(pred_y[:,1,:].cpu().data.numpy(), xform)
        imgNifti2 = nib.Nifti1Image(pred_y[:,2,:].cpu().data.numpy(), xform)
        imgNifti3 = nib.Nifti1Image(pred_y[:,3,:].cpu().data.numpy(), xform)
        imgNiftiPred = nib.Nifti1Image(segmentation_pred[:,0,:].cpu().data.numpy(), xform)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        niftiName = os.path.join(path, str(epoch) + '_1_'+img_names[0])
        niftiName2 = os.path.join(path, str(epoch) + '_2_'+ img_names[0])
        niftiName3 = os.path.join(path, str(epoch) + '_3_'+ img_names[0])
        niftiNamePred = os.path.join(path, str(epoch) + '_combined_'+ img_names[0])
        nib.save(imgNifti, niftiName)
        nib.save(imgNifti2, niftiName2)
        nib.save(imgNifti3, niftiName3)
        nib.save(imgNiftiPred, niftiNamePred)
        '''
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        t = _to_itk(np.squeeze(pred_y.data.cpu().numpy().astype('float32'))) # x is 3 dimentional. ex. (200,200,184)
        sitk.WriteImage(t, os.path.join(path, str(epoch) + img_names[0]))

        t = _to_itk(np.squeeze(Segmentation.data.cpu().numpy())) # x is 3 dimentional. ex. (200,200,184)
        sitk.WriteImage(t, os.path.join(path, str(epoch) + '_GT_' + img_names[0]))
    return

   

def inference_ResNet(netEnc, netDec, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    netEnc.eval()
    netDec.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction = netDec(netEnc(MRI))
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesN, Dices1, Dices2, Dices3= dice(segmentation_prediction, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3]
    
    
def inference_multiTask(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    voldiff = []
    xDiff = []
    yDiff = []

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction, reg_output = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation(Segmentation)
        
        # Regression
        feats = getValuesRegression(labels)
        feats_t = torch.from_numpy(feats).float()
        featsVar = to_var(feats_t)
        
        diff =  reg_output - featsVar 
        diff_np = diff.cpu().data.numpy()
        
        voldiff.append(diff_np[0][0])
        xDiff.append(diff_np[0][1])
        yDiff.append(diff_np[0][2])

                    
        DicesN, Dices1, Dices2, Dices3= dice(segmentation_prediction, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3, voldiff, xDiff, yDiff]
    
def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


def resizeTensorMask(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              numClasses,
                              img_size / scalingFactor,
                              img_size / scalingFactor))

    for i in range(data.shape[0]):

        for l in range(numClasses):
            img = data[i, l, :, :].reshape(img_size, img_size)
            imgRes = skiTransf.resize(img, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
            idx0 = np.where(imgRes < 0.5)
            idx1 = np.where(imgRes >= 0.5)
            imgRes[idx0] = 0
            imgRes[idx1] = 1
            resizedLabels[i, l, :, :] = imgRes

    tensorClass = torch.from_numpy(resizedLabels).float()
    return Variable(tensorClass.cuda())


def resizeTensorMaskInSingleImage(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              img_size / scalingFactor,
                              img_size / scalingFactor))

    for i in range(data.shape[0]):
        img = data[i, :, :].reshape(img_size, img_size)
        imgL = np.zeros((img_size, img_size))
        idx1t = np.where(img == 1)
        imgL[idx1t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx1 = np.where(imgRes >= 0.5)

        imgL = np.zeros((img_size, img_size))
        idx2t = np.where(img == 1)
        imgL[idx2t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx2 = np.where(imgRes >= 0.5)

        imgL = np.zeros((img_size, img_size))
        idx3t = np.where(img == 1)
        imgL[idx3t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx3 = np.where(imgRes >= 0.5)

        imgResized = np.zeros((img_size / scalingFactor, img_size / scalingFactor))
        imgResized[idx1] = 1
        imgResized[idx2] = 2
        imgResized[idx3] = 3

        resizedLabels[i, :, :] = imgResized

    tensorClass = torch.from_numpy(resizedLabels).long()
    return Variable(tensorClass.cuda())


# TODO : use lr_scheduler from torch.optim
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


# TODO : use lr_scheduler from torch.optim
def adjust_learning_rate(lr_args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_args * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(" --- Learning rate:  {}".format(lr))


