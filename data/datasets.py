import os
import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights, ViT_B_32_Weights, ConvNeXt_Base_Weights

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
    def __init__(self, metadata_path: str, image_dir_path: str, model_type: str, is_train: bool = True):
        """
        Dataset subclass which preprocesses deepfakes
        
        Parameters
        ----------
            metadata_path: str
                file path for metadata csv
            image_dir_path: str
                path for directory contain images
            model_type: {'ResNet', 'ViT', 'ConvNeXt'}
                type of vison model
            is_train: bool, default = True
                whether the dataset is to be used for training and development or testing
        """
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir_path = image_dir_path
        self.model_type = model_type
        self.is_train = is_train

        # load image preprocessors for pretrained models
        if self.model_type == 'ResNet':
            self.base_transforms = ResNet50_Weights.DEFAULT.transforms()
        elif self.model_type == 'ViT':
            self.base_transforms = ViT_B_32_Weights.DEFAULT.transforms()
        elif self.model_type == 'ConvNeXt':
            self.base_transforms = ConvNeXt_Base_Weights.DEFAULT.transforms()
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        image_metadata_record = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir_path, image_metadata_record['Filename'])
        image = Image.open(image_path)
        image = image.convert("RGB")

        if self.model_type not in ['ResNet', 'ViT', 'ConvNeXt']:
            # Training transforms (includes randomization to augment data for each epoch)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)), # Randomly crop image NOTE: size can be changed if not resnet / ViT
                # transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # Randomly change sharpness
                transforms.ToTensor() # Convert to tensor
            ])

            # Validation / testing transforms (does not include randomization)
            val_transform = transforms.Compose([
                transforms.Resize(232), # Resize short side (matches Resnet but can be changed)
                transforms.CenterCrop(224), # Extract center of image
                transforms.ToTensor()
            ])
        else:
            # Training transforms (includes randomization to augment data for each epoch)
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),  # Randomly crop image
                # transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # Randomly change sharpness
                self.base_transforms # This implicitly applies ToTensor and Normalize
            ])

            # Validation / testing transforms (does not include randomization)
            val_transform = self.base_transforms
        
        if self.is_train:
            image_tensor = train_transform(image)
        else:
            image_tensor = val_transform(image)

        return image_tensor, 1 if image_metadata_record['Ground Truth'] == 'Fake' else 0