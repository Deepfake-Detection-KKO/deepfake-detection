import os
import open_clip
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
if hasattr(torch.mps, "is_available") and torch.mps.is_available():
    torch.mps.manual_seed(SEED) 

class DeepFakeDataset(Dataset):
    def __init__(self, metadata_path: str, image_dir_path: str, model_type: str, is_train: bool = True, return_metadata = False, device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
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
        self.metadata['Filename'] = self.metadata['Filename'].fillna("")
        self.metadata['Public Comments'] = self.metadata['Public Comments'].fillna("")
        self.image_dir_path = image_dir_path
        self.model_type = model_type
        self.is_train = is_train
        self.return_metadata = return_metadata

        # load image preprocessors for pretrained models
        if self.model_type == 'ResNet':
            self.base_transforms = ResNet50_Weights.DEFAULT.transforms()
        elif self.model_type == 'ViT':
            self.base_transforms = ViT_B_32_Weights.DEFAULT.transforms()
        elif self.model_type == 'ConvNeXt':
            self.base_transforms = ConvNeXt_Base_Weights.DEFAULT.transforms()
        elif self.model_type == 'ResNet-CLIP':
            _, _, self.base_transforms = open_clip.create_model_and_transforms('RN50', pretrained='cc12m', device = device)
        elif self.model_type == 'ViT-CLIP':
            _, _, self.base_transforms = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device = device)
        elif self.model_type == 'ConvNeXt-CLIP':
            _, _, self.base_transforms = open_clip.create_model_and_transforms('convnext_base', pretrained='laion400m_s13b_b51k', device = device)

        if self.model_type not in ['ResNet', 'ViT', 'ConvNeXt', 'ResNet-CLIP', 'ViT-CLIP', 'ConvNeXt-CLIP']:
            # Training transforms (includes randomization to augment data for each epoch)
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)), # Randomly crop image NOTE: size can be changed if not resnet / ViT
                # transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # Randomly change sharpness
                transforms.ToTensor() # Convert to tensor
            ])

            # Validation / testing transforms (does not include randomization)
            self.val_transform = transforms.Compose([
                transforms.Resize(232), # Resize short side (matches Resnet but can be changed)
                transforms.CenterCrop(224), # Extract center of image
                transforms.ToTensor()
            ])
        else:
            # Training transforms (includes randomization to augment data for each epoch)
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),  # Randomly crop image
                # transforms.RandomAdjustSharpness(sharpness_factor=0.2, p=0.5), # Randomly change sharpness
                self.base_transforms # This implicitly applies ToTensor and Normalize
            ])

            # Validation / testing transforms (does not include randomization)
            self.val_transform = self.base_transforms
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        image_metadata_record = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir_path, image_metadata_record['Filename'])
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        if self.is_train:
            image_tensor = self.train_transform(image)
        else:
            image_tensor = self.val_transform(image)

        if self.return_metadata:
            return image_tensor, 1 if image_metadata_record['Ground Truth'] == 'Fake' else 0, image_metadata_record['Filename'], image_metadata_record['Public Comments']
        else:
            return image_tensor, 1 if image_metadata_record['Ground Truth'] == 'Fake' else 0