import os
import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
import torchvision.transforms as transforms


class DeepFakeDataset(Dataset):
    def __init__(self, metadata_path, image_dir_path, transform = None, target_transform = None):
        self.metadata = pd.read_csv(metadata_path)
        self.image_dir_path = image_dir_path
        # self.transform = transform
        # self.target_transform = target_transform
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        image_metadata_record = self.metadata.iloc[idx]
        image_path = os.path.join(self.image_dir_path, image_metadata_record['Filename'])
        image = Image.open(image_path)
        image = image.convert("RGB")
        image = transforms.Resize((400, 400))(image)
        image = transforms.ToTensor()(image)


        return image, 1 if image_metadata_record['Ground Truth'] == 'Fake' else 0