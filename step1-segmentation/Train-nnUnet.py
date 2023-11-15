import os
from torch.utils.data import DataLoader
from progressBar import printProgressBar
import medicalDataLoader128
import monai
from UNet3D import *
from utils import *
import time
import random
import numpy as np
from losses import DiceLoss
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def runTraining(num_classes = 88):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    lr = 0.01 # intial learning eate
    batch_size = 8
    epoch = 100
    batch_size_val = 8
    root_dir = '../data/' # a path to the root folder of the data
    df_root_dir = '../dataframes/' # a path to the folder of the dataframes
    modality = 'T2'
    model_name = 'nn-UNet'

    base_path = f'Results-{model_name}-128-{num_classes-1}classes/' # path to save models and statistics
    mainPath = base_path + 'Statistics/'
    model_dir = base_path + 'model/' 
    print(model_name)

    # set random seed for all gpus
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TODO: Dataloader
    train_set = medicalDataLoader128.MedicalImage3DDataset('train', root_dir, modality, num_classes, df_root_path=df_root_dir) #augment=False, #equalize=False)
    train_loader = DataLoader(train_set, batch_size=min(batch_size, len(train_set)), num_workers=8, shuffle=True, )
    val_set = medicalDataLoader128.MedicalImage3DDataset('val', root_dir, modality, num_classes, df_root_path=df_root_dir)
    val_loader = DataLoader(val_set, batch_size=min(batch_size_val,len(val_set)), num_workers=8, shuffle=False)
    print('TOTAL TRAIN IMAGES:', len(train_loader), len(train_set))
    print('TOTAL VAL IMAGES:', len(val_loader), len(val_set))
    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    
    # Load network
    spacings = [1,1,1]
    patch_size = [128, 128, 128]
    strides, kernels, sizes = [], [], patch_size[:]

    while True:
        spacing_ratio = [spacing / min(spacings) for spacing in spacings]
        stride = [
            2 if ratio <= 2 and size >= 2 * 4 else 1 for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
        if len(strides) == 5:
            break
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    netG = monai.networks.nets.DynUNet(3, 1, num_classes, kernels, strides, strides[1:])

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = DiceLoss(normalization='none')

    if torch.cuda.is_available():
        netG.cuda()
        netG = nn.DataParallel(netG)
        softMax.cuda()
        CE_loss.cuda()
        Dice_loss.cuda()
        Dice_loss.cuda()
    
        
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)
   
    BestDice, BestTotal, BestCE, BestEpoch = 1000, 1000, 1000, 0

    Losses_total = []
    Losses_CE = []
    Losses_Dice = []

    Losses_val_total = []
    Losses_val_CE = []
    Losses_val_Dice = []

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        labelled_batches = len(train_loader)
        val_batches = len(val_loader)
        loss_total = []
        loss_Dice = []
        loss_CE = []
        
        timesAll = []
        print('-' * 40) 
        start_time = time.time()
        
        for j, data in enumerate(train_loader):
            if j%2==0:
                pass
            
            image = data[0]
            mask = data[1]
            labels = data[2]
            img_names = data[3]

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            netG.train()
            optimizerG.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)
            
            ################### Train ###################
            netG.zero_grad()

            segmentation_prediction = netG(MRI)
            #To softmax
            predClass_y = softMax(segmentation_prediction)
            
            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)

            CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)

            # OneHot for Dice
            Segmentation_planes = getOneHotSegmentation(Segmentation, num_classes)
            #Dice loss
            Dice_lossG = Dice_loss(predClass_y, Segmentation_planes)

            lossG = CE_lossG + Dice_lossG


            lossG.backward()
            optimizerG.step()
            if not(lossG.cpu().data.numpy() > 0 and lossG.cpu().data.numpy()<10):
                print('*************')
                print(img_names)
                print('*************')
                break
            # Save for plots
            loss_total.append(lossG.cpu().data.numpy())
            loss_CE.append(CE_lossG.cpu().data.numpy())
            loss_Dice.append(Dice_lossG.cpu().data.numpy())

            printProgressBar(j + 1, labelled_batches,
                             prefix="[Labelled Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" loss_total: {:.4f},  loss_CE: {:.4f},  loss_Dice: {:.4f}".format(lossG.data, CE_lossG.data, Dice_lossG.data))
  
        print('')

        
        directory = mainPath
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, 'train-Losses.npy'), Losses_total)
        np.save(os.path.join(directory, 'train-Losses_CE.npy'), Losses_CE)
        np.save(os.path.join(directory, 'train-Losses_Dice.npy'), Losses_Dice)
    

        loss_val_total = []
        loss_val_Dice = []
        loss_val_CE = []
        torch.cuda.empty_cache()
        for j, data in enumerate(val_loader):
            with torch.no_grad():
                image = data[0]
                labels = data[2]
                if image.size(0) != batch_size:
                    continue

                netG.eval()
                optimizerG.zero_grad()
                MRI = to_var(image)
                Segmentation = to_var(labels)
                ################### Train ###################
                netG.zero_grad()
                segmentation_prediction = netG(MRI)
                segmentation_prediction = netG(MRI)

                predClass_y = softMax(segmentation_prediction)
                Segmentation_class = getTargetSegmentation(Segmentation)
                CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)
                Segmentation_planes = getOneHotSegmentation(Segmentation, num_classes)
                Dice_lossG = Dice_loss(predClass_y, Segmentation_planes)
                lossG = CE_lossG + Dice_lossG

                # Save for plots
                loss_val_total.append(lossG.cpu().data.numpy())
                loss_val_CE.append(CE_lossG.cpu().data.numpy())
                loss_val_Dice.append(Dice_lossG.cpu().data.numpy())

            printProgressBar(j + 1, val_batches,
                             prefix="[validation] Epoch: {} ".format(i),
                             length=15,
                             suffix=" loss_val_total: {:.4f},  loss_val_CE: {:.4f},  loss_val_Dice: {:.4f}".format(lossG.data, CE_lossG.data, Dice_lossG.data))


        Losses_val_total.append(np.mean(loss_val_total))
        Losses_val_CE.append(np.mean(loss_val_CE))
        Losses_val_Dice.append(np.mean(loss_val_Dice))

        np.save(os.path.join(directory, 'val-Losses.npy'), Losses_val_total)
        np.save(os.path.join(directory, 'val-Losses_CE.npy'), Losses_val_CE)
        np.save(os.path.join(directory, 'val-Losses_Dice.npy'), Losses_val_Dice)

        printProgressBar(labelled_batches, labelled_batches,
                done="[Training] Epoch: {}, Loss_total: {:.4f}, Loss_CE: {:.4f}, Loss_Dice: {:.4f}".format(
                    i, np.mean(loss_val_total), np.mean(loss_val_CE), np.mean(loss_val_Dice)))

        CurrentDice = np.mean(loss_val_Dice)
        print()
        if CurrentDice < BestDice:
            BestLoss = CurrentDice
            BestEpoch = i
            BestTotal = np.mean(loss_val_Dice)
            BestCE = np.mean(loss_val_CE)
            BestDice = np.mean(loss_val_Dice)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(netG, os.path.join(model_dir, "Best_" + model_name + ".pkl"))


        print("###                                                                                  ###")
        print("### Best Dice Loss(mean): {:.4f} at epoch {} with (Dice loss): {:.4f}  (Total Loss): {:.4f} (CE Loss): {:.4f} ###".format(BestDice, BestEpoch, BestDice, BestTotal, BestCE))
        print("###                                                                                  ###")
        print('Time spent for entire epoch:  {:.4f}'.format(time.time() -start_time))
        print(' ')
        #spentTime = time.time()-start_time
        
        # # This is not as we did it in the MedPhys paper
        if i % 20 == 19 :
            for param_group in optimizerG.param_groups:
                param_group['lr'] = lr/4

if __name__ == '__main__':
    runTraining( num_classes = 88)

