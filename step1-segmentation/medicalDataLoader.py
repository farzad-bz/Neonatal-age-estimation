from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import nibabel as nib
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

def make_3Ddataset(root_dir, df_root_path, mode, modality, num_classes):
    """
    Args:
        root_dir : path to volumes 
        df_root_path (string): dataframe directory containing csv files
        mode (string): 'train', 'val', 'test'
        num_classes : 10 or 88 for DHCP data
    """
    assert mode in ['train', 'val', 'test']
    items = []
    
    df = pd.read_csv(f'{df_root_path}/DHCP-patches-128-{mode}-dataframe.csv').reset_index(drop=True)
    df = df.query(f'{modality}_modality == {modality}_modality') #those who have selected modality data

    data_paths = df[f'{modality}_modality'].values
    GT_paths = df[f'segmentation_{num_classes-1}classes'].values
    mask_paths = df['brainmask_bet'].values
    
    subs = df['sub'].values
    sess = df['ses'].values
    
    names = [subs[i]+'-'+str(sess[i]) for i in range(len(df))]

    patches = df['patches'].values

    for it_im, it_mk, it_gt, it_patch, it_nm in zip(data_paths, mask_paths, GT_paths, patches, names):
        item = (os.path.join(root_dir, it_im), os.path.join(root_dir, it_mk), os.path.join(root_dir, it_gt), eval(it_patch), it_nm)
        items.append(item)

    return items 



def normalize_intensity(img_tensor, normalization="mean", norm_values=(0, 1, 1, 0)):
    """
    Accept the image tensor and normalizes it (ref: MedicalZooPytorch)
    Args: 
        img_tensor (tensor): image tensor
        normalization (string): choices = "max", "mean"
        norm_values (array): (MEAN, STD, MAX, MIN)

    """
    if normalization == "mean":
        mask = img_tensor.ne(0.0)
        desired = img_tensor[mask]
        mean_val, std_val = desired.mean(), desired.std()
        img_tensor = (img_tensor - mean_val) / (std_val + 1e-10)
    elif normalization == "max":
        #max_val, _ = torch.max(img_tensor)
        #img_tensor = img_tensor / max_val
        img_tensor = img_tensor/img_tensor()
    elif normalization == 'brats':
        # print(norm_values)
        normalized_tensor = (img_tensor.clone() - norm_values[0]) / norm_values[1]
        final_tensor = torch.where(img_tensor == 0., img_tensor, normalized_tensor)
        final_tensor = 100.0 * ((final_tensor.clone() - norm_values[3]) / (norm_values[2] - norm_values[3])) + 10.0
        x = torch.where(img_tensor == 0., img_tensor, final_tensor)
        return x


    elif normalization == 'max_min':
        #img_tensor = (img_tensor - norm_values[3]) / ((norm_values[2] - norm_values[3]))
        img_tensor = img_tensor - img_tensor.min()
        img_tensor = img_tensor/img_tensor.max()

    elif normalization == None:
        img_tensor = img_tensor
    return img_tensor


class MedicalImage3DDataset(Dataset):
    """DHCP-r2 dataset."""

    def __init__(self, mode, root_dir, modality, num_classes, normalization='mean', df_root_path = '.', labelled_if_test = False, seed = 0, labelled_samples=0):
        """
        Args:
            mode: 'train','val','test'
            root_dir (string): Directory with all the volumes.
            transform (callable, optional): Optional transform to be applied on a sample.
            df_root_path (string): dataframe directory containing csv files
        """
        self.normalization = normalization
        self.num_classes = num_classes
        self.root_dir = root_dir
        self.mode = mode
        self.imgs = make_3Ddataset(root_dir, df_root_path, mode, modality, num_classes)



    def transform_volume(self, x):
        if len(x.shape) == 3:
            x = np.expand_dims(x, -1)
        x = torch.from_numpy(x.transpose((-1, 0 , 1 , 2)))
        return x

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        data_path, mask_path, GT_path, patch, name = self.imgs[index]
        img = nib.load(data_path).get_fdata(dtype=np.float32)
        mask = nib.load(mask_path).get_fdata(dtype=np.float32)
        gt = nib.load(GT_path).get_fdata(dtype=np.float32)
        img[mask==0] = 0
        while gt.sum()<20:
            index = np.random.randint(len(self.imgs))
            data_path, mask_path, GT_path, patch, name = self.imgs[index]
            img = nib.load(data_path).get_fdata(dtype=np.float32)
            mask = nib.load(mask_path).get_fdata(dtype=np.float32)
            gt = nib.load(GT_path).get_fdata(dtype=np.float32)
            img[mask==0] = 0
        if self.num_classes == 10:
            gt[gt==4] = 0
        elif self.num_classes == 88:
            gt[gt==84] = 0
        
        if self.mode == 'train':
            img = img * np.random.uniform(0.6, 1.4)

        img = self.transform_volume(img)

        # Normalization
        '''
        MEAN, STD, MAX, MIN = 0., 1., 1., 0.
        MEAN, STD = img_FLAIR.mean(), img_FLAIR.std()
        MAX, MIN = img_FLAIR.max(), img_FLAIR.min()
        img_FLAIR = normalize_intensity(img_FLAIR, normalization=normalization, norm_values=(MEAN, STD, MAX, MIN))
        '''
        #img = normalize_intensity(img, normalization='mean')
    
        #mask = mask.unsqueeze(dim=1)
        #print(img_T1w.shape, mask.shape)
        #print([img_T1w.max(), img_T1w.min(), img_T1w.mean()])
        #print([np.max(mask), np.min(mask), np.mean(mask)])

        return [img, mask, gt, name]
