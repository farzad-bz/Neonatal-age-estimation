# Neonatal-age-estimation
Codes of the paper "Determining regional brain growth in premature and mature infants in relation to age at MRI using deep neural networks"


### Step 0: Preprocessing the data
In the initial stage, it is important to standardize the resolution of all MRIs to a consistent size, such as 0.50x50x50 This essential preprocessing step significantly affects the extracted features and the segmentations in the next steps. Also, the dataset should be split into training, validation, and testing sets. Then, to train the segmentation network better and decrease the GPU required for training the model, we extract 128x128x128 patches and save them as well as their corresponding dataframe. An example of patches-dataframe can be seen in the "dataframes" folder


### step1: Training the segmentation network
Using "Train-nnUnet.py" train the segmentation network (nn-UNet) with your training data. Please bear in mind that you should modify the dataloader and the codes based on your input. DHCP dataset [Makropoulos A. et al. TMI 2014] had been originally used for the paper and the codes are based on DHCP NIFTI files which are initially segmented using DRAW-EM.


### step2: Extracting Regions' Volume and Surface-to-Volume Ratio features.
Using "Extract_features.py" extract the surface-to-volume ratio and relational volume for each region of the brain (each label). This part of the code will save a data frame including all extracted features for all subjects. 


### step3: training Bayesian Ridge regression model for age estimation regression
In the final step, using "Age_estimation.py" a Bayesian ridge regression model will be trained based on the extracted features of the training set from previous steps. Then for the test set, it compares the predicted age results with their actual age and reports the R^2 and MAE metrics.

