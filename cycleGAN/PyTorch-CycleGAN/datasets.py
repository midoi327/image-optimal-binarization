import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned # True
        
        if mode == 'train':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A_resized' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B_fid300' % mode) + '/*.*'))
        elif mode == 'test':
            self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
            self.files_B = sorted(glob.glob(os.path.join(root, '%s/B_fid300' % mode) + '/*.*'))
            

    def __getitem__(self, index):
        
        
        # print("Original image shape:", np.array(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')).shape)
        
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))
        
        # print("Transformed image shape:", item_A.shape)
        
        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))

        # print('A:', np.array(item_A).shape, '\nB:', np.array(item_B).shape) #(3, 256, 256)
        
        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))