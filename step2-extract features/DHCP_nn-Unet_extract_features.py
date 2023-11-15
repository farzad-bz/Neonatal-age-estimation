
import pandas as pd
import numpy as np
import nibabel as nib
import os
import torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def getOneHotSegmentation(batch, num_classes):

    labels = [i for i in range(num_classes)]


    batch = batch.unsqueeze(dim=0)
    oneHotLabels = torch.cat(tuple([batch == i for i in labels]), dim=0)

    return oneHotLabels.float()


train_df = pd.read_csv('../dataframes/DHCP-train-dataframe.csv', index_col=0)
val_df = pd.read_csv('../dataframes/DHCP-val-dataframe.csv', index_col=0)
test_df = pd.read_csv('../dataframes/DHCP-test-dataframe.csv', index_col=0)


m = torch.nn.AvgPool3d((3, 3, 3), stride=(1, 1, 1)).cuda()

for mode in ['train', 'val', 'test']:
    subs = []
    sess = []
    birth_ages = []
    scan_ages = []
    genders = []
    birth_weights = []
    head_circumferences = []
    seg87_S2V_dicts = []
    seg87_RV_dicts = []
    for i,row in eval(f'{mode}_df').iterrows():
        sub = row['sub']
        ses = row['ses']
        subs.append(sub)
        sess.append(ses)
        seg87_S2V = {} #surface to volume ratio (ratio of region's surface (boundry voxel) to region's volume)
        seg87_RV = {}  #Relational volume (ratio of region's volume to the brain volume)
        seg87_path = os.path.join('Results-nn-UNet-128-87classes', 'predictions', f'sub-{sub}_ses-{ses}]_prediction.nii.gz')
        seg87 = nib.load(seg87_path).get_fdata(dtype=np.float32)

        brain_size_87 = np.sum(seg87!=0)
        seg87_onehot = getOneHotSegmentation(torch.from_numpy(seg87),  88)
        input_data = seg87_onehot.cuda()
        output = m(input_data).cpu().numpy() #using average pool to obtain boundry voxels
        for j in range(1, 88):
            if j==84:
                continue
            structure_size = np.sum(seg87_onehot[j].numpy())
            seg87_RV[j] = structure_size / brain_size_87
            surface_size = structure_size - np.sum(output[j]==1)
            seg87_S2V[j] = surface_size/structure_size
        
        seg87_S2V_dicts.append(seg87_S2V)
        seg87_RV_dicts.append(seg87_RV)
        subs.append(sub)
        sess.append(ses)
        birth_ages.append(row['birth_age'])
        scan_ages.append(row['scan_age'])
        genders.append(row['gender'])
        birth_weights.append(row['birth_weight'])
        head_circumferences.append(row['scan_head_circumference'])

    ouput_df = pd.DataFrame({'sub':subs, #create dataframe from extracted features
            'ses':sess, 
            'gender':genders,
            'birth_age':birth_ages, 
            'scan_age':scan_ages, 
            'head_circumference':head_circumferences, 
            'birth_weight':birth_weights,
            'seg87_relational_volume':seg87_RV_dicts, 
            'seg87_surface_to_volume_ratio':seg87_S2V_dicts})
    ouput_df.to_csv(f'DHCP_{mode}_nn-UNet_extracted_features.csv')
