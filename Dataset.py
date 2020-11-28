from PIL import Image
import torch 
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os

torch.manual_seed(0)

class HistoCancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type = "train"):

        #path to images
        path2data = os.path.join(data_dir, data_type)

        #get a list of images
        filenames = os.listdir(path2data)

        #get the full path to images
        self.full_filenames = [os.path.join(path2data, f) for f in filenames]

        csv_filename = data_type+"_labels.csv"
        path2csvLabels=os.path.join(data_dir,csv_filename)
        labels_df=pd.read_csv(path2csvLabels)
        labels_df.set_index("id", inplace=True)
        self.labels = [labels_df.loc[filename[:-4]].values[0] for filename in filenames]

        self.transform = transform
    
    def __len__(self):

        return len(self.full_filenames)

    def __getitem__(self, idx):

        #open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])
        image = self.transform(image)
        return image, self.labels[idx]
