import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WallDataset(Dataset):
    """Dataset for wall segmentation"""
    def __init__(self, image_dir, mask_dir, transform=None, image_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        self.images = os.listdir(image_dir)
        
        # Default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load mask (assuming same filename with different extension or _mask suffix)
        # First try with _mask suffix
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            base_name = img_name.rsplit('.', 1)[0]
            mask_name = base_name + '_mask.png'
        else:
            mask_name = img_name.replace('.jpg', '_mask.png').replace('.png', '_mask.png')
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            # If mask not found, create a dummy mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to binary (0 or 1) and ensure it's float
        if isinstance(mask, torch.Tensor):
            mask = mask.float()
        else:
            mask = ((mask > 127).astype(np.float32))
            mask = torch.from_numpy(mask)
        
        return image, mask.unsqueeze(0)