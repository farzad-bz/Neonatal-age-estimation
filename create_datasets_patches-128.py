import nibabel as nib
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm


train_df = pd.read_csv('DHCP-train-dataframe.csv')
val_df = pd.read_csv('DHCP-val-dataframe.csv')
test_df = pd.read_csv('DHCP-test-dataframe.csv')

print('splitting to 3d patches started:')

root_dir = '../DHCP-R2/pruned DHCP - release 2'
stride = 32
size = 128

print(f'size of the patches: {size}')
print(f'stride of the patches: {stride}')

for mode in ['train', 'val']:
    print(mode)
    all_patches = []
    img_t1_patches = []
    img_t2_patches = []
    seg9_patches = []
    mask_patches = []
    seg87_patches = []
    subs = []
    sess = []
    for i,row in tqdm(eval(f'{mode}_df').iterrows()):
        mask = nib.load(os.path.join(root_dir, row['brainmask_bet'])).get_fdata(dtype=np.float32)
        mask = (mask).astype(np.uint8)
        mask = np.pad(mask, (0,64))

        img_t2 = nib.load(os.path.join(root_dir, row['T2_modality'])).get_fdata(dtype=np.float32)
        #img_t2 = (img_t2).astype(np.float16)
        img_t2 = np.pad(img_t2, (0,64))
        img_t2 = img_t2 / img_t2.max()

        seg9 = nib.load(os.path.join(root_dir, row['segmentation_9classes'])).get_fdata(dtype=np.float32)
        seg9 = (seg9).astype(np.uint8)
        seg9 = np.pad(seg9, (0,64))
        
        seg87 = nib.load(os.path.join(root_dir, row['segmentation_87classes'])).get_fdata(dtype=np.float32)
        seg87 = (seg87).astype(np.uint8)
        seg87 = np.pad(seg87, (0,64))

        # if type(row['T1_modality'])==str:
        #     img_t1 = nib.load(os.path.join(root_dir, row['T1_modality'])).get_fdata(dtype=np.float32)
        #     #img_t1 = img_t1.astype(np.float16)
        #     img_t1 = np.pad(img_t1, (0,32))

        sub = row['sub']
        ses = row['ses']

        min_W, max_W = (np.where(mask==1)[0].min(), np.where(mask==1)[0].max())
        min_H, max_H = (np.where(mask==1)[1].min(), np.where(mask==1)[1].max())
        min_D, max_D = (np.where(mask==1)[2].min(), np.where(mask==1)[2].max())
        
        start0_W = max(0, min_W - np.random.randint(16))
        start0_H = max(0, min_H - np.random.randint(16))
        start0_D = max(0, min_D - np.random.randint(16))
        
        start_W = start0_W
        start_H = start0_H
        start_D = start0_D

        end_W = start_W + size
        end_H = start_H + size
        end_D = start_D + size

        max_size_W = mask.shape[0]
        max_size_H = mask.shape[1]
        max_size_D = mask.shape[2]

        k = -1
        while end_W < max_W + stride:
            while end_H < max_H + stride:
                while end_D < max_D + stride:
                    if mask[start_W:end_W , start_H:end_H , start_D:end_D].max()>0.0:
                        k += 1
                        all_patches.append({'W_s':start_W, 'W_e':end_W , 'H_s':start_H, 'H_e':end_H , 'D_s':start_D, 'D_e':end_D})
                        
                        img_t2_patch = img_t2[start_W:end_W , start_H:end_H , start_D:end_D]
                        img_t2_patches.append(os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-T2-data-{k}.nii.gz'))
                        # np.save(os.path.join('DHCP-patches-128', f'{sub}-{ses}-T2-data-{k}.npy'), img_t2_patch) 
                        ni_img_t2 = nib.Nifti1Image(img_t2_patch, None)
                        nib.save(ni_img_t2, os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-T2-data-{k}.nii.gz'))
                        

                        seg9_patch = seg9[start_W:end_W , start_H:end_H , start_D:end_D]
                        seg9_patches.append(os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-segmentation-9class-{k}.nii.gz'))
                        # np.save(os.path.join('DHCP-patches-128', f'{sub}-{ses}-segmentation-9class-{k}.npy'), seg9_patch)
                        ni_img_seg9 = nib.Nifti1Image(seg9_patch, None)
                        nib.save(ni_img_seg9, os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-segmentation-9class-{k}.nii.gz'))

                        seg87_patch = seg87[start_W:end_W , start_H:end_H , start_D:end_D]
                        seg87_patches.append(os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-segmentation-87class-{k}.nii.gz'))
                        #np.save(os.path.join('DHCP-patches-128', f'{sub}-{ses}-segmentation-87class-{k}.npy'), seg87_patch)  
                        ni_img_seg87 = nib.Nifti1Image(seg87_patch, None)
                        nib.save(ni_img_seg87, os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-segmentation-87class-{k}.nii.gz'))

                        mask_patch = mask[start_W:end_W , start_H:end_H , start_D:end_D]
                        mask_patches.append(os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-brain-mask-{k}.nii.gz'))
                        # np.save(os.path.join('DHCP-patches-128', f'{sub}-{ses}-brain-mask-{k}.npy'), mask_patch)
                        ni_img_mask = nib.Nifti1Image(mask_patch, None)
                        nib.save(ni_img_mask, os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-brain-mask-{k}.nii.gz'))
                        
                        subs.append(sub)
                        sess.append(ses)
                        # if type(row['T1_modality'])==str:
                        #     img_t1_patch = img_t1[start_W:end_W , start_H:end_H , start_D:end_D]
                        #     img_t1_patches.append(os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-T1-data-{k}.nii.gz'))
                        #     # np.save(os.path.join('DHCP-patches-128', f'{sub}-{ses}-T2-data-{k}.npy'), img_t1_patch) 
                        #     ni_img_t1 = nib.Nifti1Image(img_t1_patch, None)
                        #     nib.save(ni_img_t1, os.path.join('DHCP-patches-128-nifti', mode, f'{sub}-{ses}-T1-data-{k}.nii.gz'))
                        # else:
                        #     img_t1_patches.append(np.nan)
                        
                    start_D += stride   
                    end_D = start_D + size
                start_D = start0_D
                end_D = start_D + size
                start_H += stride
                end_H = start_H + size
            start_H = start0_H
            end_H = start_H + size
            start_W += stride
            end_W = start_W + size

    print('saving dataframe ... ')
    df = pd.DataFrame({'sub':subs, 'ses':sess, 'T2_modality':img_t2_patches, 'brainmask_bet':mask_patches, 'segmentation_9classes':seg9_patches, 'segmentation_87classes':seg87_patches, 'patches':all_patches})
    df.to_csv(f'DHCP-patches-128-{mode}-dataframe.csv')
    print('data frame saved')