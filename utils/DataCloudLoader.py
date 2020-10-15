import os
import glob

import pandas as pd


class DataCloudLoader():
    def __init__(self, rootDataDir = "/data.local/tuannm/ML/understanding_cloud_organization/data"):
        self.rootDataDir = rootDataDir
        pass
    
    def load(self):
        print("Loading DATA Cloud")
        pass
    
    
    
    
    def dataOverivew(self):
        
        print(os.listdir(self.rootDataDir))
        trainData = glob.glob(os.path.join(self.rootDataDir, "train_images", "*.jpg*"))
        print("#images: {} in train dataset".format(len(trainData)))
        
        
        testData = glob.glob(os.path.join(self.rootDataDir, "test_images", "*.jpg*"))
        print("#images: {} in test dataset".format(len(testData)))
        
        
        trainCSV = pd.read_csv(os.path.join(self.rootDataDir, "train.csv")) # Image_Label (0011165.jpg_Fish), EncodedPixels
        print(trainCSV.head())
        
        print(trainCSV['Image_Label'].apply(lambda x: x.split('_')[1]).value_counts())
        
        # label have value in Encoded Pixels
        print(trainCSV.loc[trainCSV['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[1]).value_counts())
        print(trainCSV.loc[trainCSV['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts())
#             b01c91f.jpg    4
#             c7d28ff.jpg    4
#             65144b5.jpg    4
#             eba901f.jpg    4
#             be7fe4e.jpg    4
#                           ..
#             3b69197.jpg    1
#             61c707f.jpg    1
#             144d4a8.jpg    1
#             c20fc81.jpg    1
#             e36c7a3.jpg    1

        print(trainCSV.loc[trainCSV['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().value_counts())
#         2    2372
#         3    1560
#         1    1348
#         4     266  266 images have all four masks.

        # value_counts() dau tien la count xem moi anh co bao nhieu mask, value_count() thu 2 la count: co bao nhieu anh xuat hien k masks
        
        # map image id with label
        trainCSV['label'] = trainCSV['Image_Label'].apply(lambda x: x.split('_')[1])
        trainCSV['im_id'] = trainCSV['Image_Label'].apply(lambda x: x.split('_')[0])
        

        return trainCSV
    

data = DataCloudLoader()
data.dataOverivew()

