import torch
from torch.utils.data import Dataset

"""

Module contains several pytorch datasets.

Author: Maria J. Molina, NCAR (molina@ucar.edu)

"""

class CustomDataset(Dataset):
    
    def __init__(self, traindata, testdata, transform=None, target_transform=None):
        
        """Minimal dataset preprocessing for training."""
        
        self.img_train = traindata
        self.img_label = testdata
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        
        return len(self.img_label)

    def __getitem__(self, idx):
        
        image = self.img_train[idx]
        label = self.img_label[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return {'train': image, 'test': label, 'minibatch_indx': idx}
    
    
class CustomLSTMDataset(Dataset):
    
    def __init__(self, traindata, testdata, transform=None, target_transform=None):
        
        """Minimal dataset preprocessing for training."""
        
        self.img_train = traindata
        self.img_label = testdata
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        
        return len(self.img_label)

    def __getitem__(self, idx):
        
        image = self.img_train[:, idx]
        label = self.img_label[:, idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
            
        return {'train': image, 'test': label}
