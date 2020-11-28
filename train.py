import torchvision.transforms as transforms
import torch
from Dataset import HistoCancerDataset
from torch.utils.data import random_split


data_transformer = transforms.Compose([transforms.ToTensor()])


data_dir = "../data/histopathologic-cancer-detection/"

histo_dataset = HistoCancerDataset(data_dir, data_transformer)

len_histo = len(histo_dataset)
len_train = int(0.8*len_histo)
len_val = len_histo - len_train
train_ds,val_ds=random_split(histo_dataset,[len_train,len_val])

print("train dataset length:", len(train_ds))
print("validation dataset length:", len(val_ds))