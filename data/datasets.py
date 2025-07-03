import os
import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, ViT_B_32_Weights
import random
import numpy as np

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
if torch.mps.is_available():
    torch.mps.manual_seed(SEED) 

class DeepFakeDataset(Dataset):
    def __init__(self, metadata_path, image_dir_path, model_type, is_train, transform = None, target_transform = None):
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir_path = image_dir_path
        self.model_type = model_type
        self.is_train = is_train
        # self.transform = transform
        # self.target_transform = target_transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        image_metadata_record = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir_path, image_metadata_record['Filename'])
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.model_type not in ['ResNet', 'ViT']:
            # Training transforms (includes randomization to augment data for each epoch)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)), # Randomly crop image NOTE: size can be changed if not resnet / ViT
                transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # Randomly change sharpness
                transforms.ToTensor() # Convert to tensor
            ])

            # Validation / testing transforms (does not include randomization)
            val_transform = transforms.Compose([
                transforms.Resize(232), # Resize short side (matches Resnet but can be changed)
                transforms.CenterCrop(224), # Extract center of image
                transforms.ToTensor()
            ])
        else:
            if self.model_type == 'ResNet': # ResNet default transformation
                weights = ResNet50_Weights.DEFAULT 
            elif self.model_type == 'ViT': # ViT default transformation
                weights = ViT_B_32_Weights

            base_transforms = weights.transforms() # Base transform from pretrained model, includes ToTensor and Normalize

            # Training transforms (includes randomization to augment data for each epoch)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),  # Randomly crop image
                transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # Randomly change sharpness
                base_transforms # This implicitly applies ToTensor and Normalize
            ])

            # Validation / testing transforms (does not include randomization)
            val_transform = base_transforms
        
        if self.is_train:
            image_tensor = train_transform(image)
        else:
            image_tensor = val_transform(image)

        return image_tensor, 1 if image_metadata_record['Ground Truth'] == 'Fake' else 0