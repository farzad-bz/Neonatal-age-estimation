
from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import medpy
import pandas as pd
from tqdm import tqdm
import medicalDataLoader
from UNet import *
from utils import *
import sys

import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print('-' * 40)
print('~~~~~~~~  Starting the inference... ~~~~~~')
print('-' * 40)



def prepare_img(x):
    x = np.expand_dims(x, -1)
    img_tensor = torch.from_numpy(x.transpose((-1, 0 , 1 , 2)))
    return img_tensor.unsqueeze(0)

df = pd.read_csv('../dataframes/DHCP-train-dataframe.csv', index_col=0)
df.append(pd.read_csv('../dataframes/DHCP-val-dataframe.csv', index_col=0))
df.append(pd.read_csv('../dataframes/DHCP-test-dataframe.csv', index_col=0))

root = '../data/'
num_classes = 88
model_dir = f'Results-nn-UNet-128-{num_classes-1}classes' 
predictions_dir = f'{model_dir}/predictions'
if not os.path.exists(predictions_dir):
    os.makedirs(predictions_dir)

print('...Loading model...')
try:
    netG = torch.load(os.path.join(model_dir, 'model' ,"Best_nn-UNet.pkl"))

except:
    raise('--------model not restored--------')

softMax = nn.Softmax()
netG.cuda()
netG.eval()
print('--------model restored--------')


dc_metrics = []
dc_metrics_with_bg = []
dc_metrics_with_mask = []
prediction_paths = []
ids = []
sources = []
for i,row in tqdm(df.iterrows()):
    img = nib.load(root + row[f'T2_modality']).get_fdata(dtype=np.float32)
    img = img/img.max()

    mask = nib.load(root + row['brainmask_bet']).get_fdata(dtype=np.float32)
    gt = nib.load(root + row[f'segmentation_{num_classes-1}classes']).get_fdata(dtype=np.float32)
    
    gt = np.pad(gt, (0,64))
    gt[gt==84] = 0
    img = np.pad(img, (0,64))
    mask = np.pad(mask, (0,64)) 

    full_prediction = np.zeros((num_classes, img.shape[0], img.shape[1], img.shape[2]))
    mask_prediction = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    
    for xx in range(0, 256, 64):
        for yy in range(0, 256, 64):
            for zz in range(0, 256, 64):
                with torch.no_grad():
                    img_patch = img[xx:xx+128, yy:yy+128, zz:zz+128].copy()
                    gt_patch = gt[xx:xx+128, yy:yy+128, zz:zz+128].copy()
                    img_patch[gt_patch==0] = 0
                    img_patch = prepare_img(img_patch)
                    img_patch = img_patch
                    img_patch = to_var(img_patch)
                    
                    segmentation_prediction = netG(img_patch)
                    #To softmax
                    predClass_y = softMax(segmentation_prediction)
                    
                    segmentation_prediction_ones = predToSegmentation(predClass_y)
                    full_prediction[:, xx:xx+128, yy:yy+128, zz:zz+128] +=  predClass_y.squeeze(0).cpu().detach().numpy()
                    mask_prediction[xx:xx+128, yy:yy+128, zz:zz+128] =  1
    
    full_prediction = full_prediction[:, :-64, :-64, :-64]
    mask_prediction = mask_prediction[:-64, :-64, :-64]
    mask = mask[:-64, :-64, :-64]
    gt = gt[:-64, :-64, :-64]

    segmentation_gt_ones = getOneHotSegmentation(torch.Tensor(gt).unsqueeze(0), num_classes).squeeze(0).detach().numpy()
    segmentation_prediction_ones = predToSegmentation(torch.Tensor(full_prediction).unsqueeze(0)).detach().numpy()
    binary_dcs = []

    for j in range(num_classes):
        if j==84 or j==0: ##skull and background labels
            continue
        dc = medpy.metric.binary.dc(segmentation_prediction_ones[0][j], segmentation_gt_ones[j])
        binary_dcs.append(dc)

    mask_prediction[mask==0.0] = 1
    full_prediction[0][mask==0.0] = 1000 
    segmentation_prediction_ones = predToSegmentation(torch.Tensor(full_prediction).unsqueeze(0)).detach().numpy()
    binary_dcs_with_mask = []  
    for j in range(num_classes):
        if j==84 or j==0: ##skull and background labels
            continue
        dc = medpy.metric.binary.dc(segmentation_prediction_ones[0][j], segmentation_gt_ones[j])
        binary_dcs_with_mask.append(dc)

    dc_metrics.append(np.mean(binary_dcs[1:]))

    dc_metrics_with_bg.append(np.mean(binary_dcs))

    dc_metrics_with_mask.append(np.mean(binary_dcs_with_mask[1:]))

    output = segmentation_prediction_ones[0].argmax(0)
    ni_output = nib.Nifti1Image(output.astype(np.uint8), None)
    nib.save(ni_output, os.path.join(predictions_dir, f'sub-{row["sub"]}_ses-{row["ses"]}]_prediction.nii.gz'))
prediction_df = pd.DataFrame({'id':ids,
'dc_metrics' : dc_metrics, 
'dc_metrics_with_bg' : dc_metrics_with_bg, 
'dc_metrics_with_mask' : dc_metrics_with_mask, 
})
prediction_df.to_csv(f'{model_dir}'+ f'/Statistics/metrics.csv')
