import torch
from torch.utils.data import Dataset

"""

Module contains pytorch dataset.

Author: Maria J. Molina, UMD (mjmolina@umd.edu)

"""

class CustomDataset(Dataset):
    
    def __init__(self, data_input, data_label):
        
        """
        Minimal pytorch dataset. No frills.
        
        Args:
            data_input (array): input images.
            data_label (array): input labels.
        """
        
        self.img_train = data_input
        self.img_label = data_label
        

    def __len__(self):
        
        return len(self.img_label)

    def __getitem__(self, idx):
        
        image = self.img_train[idx]
        label = self.img_label[idx]
            
        return {'input': image, 'label': label}
