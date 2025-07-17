import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepfake_utils.datasets import DeepFakeDataset
from deepfake_utils.train import train
from deepfake_utils.models import MyModel
import random
import time

# Constants
BATCH_SIZE = 64
NUM_WORKERS = 4

# Set random seed
SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
if torch.mps.is_available():
    torch.mps.manual_seed(SEED)

# Paths
IMAGE_DIR_PATH = 'Deepfake-Eval-2024/image-data'

# Device
device = torch.accelerator.current_accelerator()
print(f'Using {device} accelerator \n')

# Loss function
loss_fn = nn.CrossEntropyLoss(reduction = 'sum')

# Hyperparameters TODO update
model_types = ['ViT-b32-pretrained']
dropout_rates = [0.2]
learning_rates = [1e-3, 1e-4]
epochs_list = [10]
use_lr_scheduler = True

experiment_id = 1

# Iterate through hyperparameters
for model_type in model_types:
    if "ViT" in model_type:
        transform_type = 'ViT'
    elif "ResNet" in model_type:
        transform_type = 'ResNet'
    
    # Load data
    train_data = DeepFakeDataset("image-metadata-train.csv", IMAGE_DIR_PATH, transform_type, is_train = True)
    train_data_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=NUM_WORKERS)

    val_data = DeepFakeDataset("image-metadata-val.csv", IMAGE_DIR_PATH, transform_type, is_train = False)
    val_data_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = False, num_workers=NUM_WORKERS)
    
    for dropout_rate in dropout_rates:
        for epochs in epochs_list:
            for learning_rate in learning_rates:
                start_time = time.time()

                # Create model
                model = MyModel(
                    model_type=model_type,
                    device=device,
                    dropout_rate=dropout_rate
                )

                # Optimizer and learning rate schedule
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                if use_lr_scheduler:
                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
                    )
                
                # Train the model
                print(f'Experiment {experiment_id}, model type: {model_type}')
                train_loss_history, train_roc_auc_history, train_pr_auc_history, train_acc_history, val_loss_history, val_roc_auc_history, val_pr_auc_history, val_acc_history = train(epochs, train_data_loader, val_data_loader, model, loss_fn, optimizer, device)
                experiment_record = {
                    'experiment_id': experiment_id,
                    'model': model_type,
                    'dropout_rate': dropout_rate,
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'train_loss_history': [train_loss_history], 
                    'train_roc_auc_history': [train_roc_auc_history],
                    'train_pr_auc_history': [train_pr_auc_history], 
                    'train_acc_history': [train_acc_history], 
                    'val_loss_history': [val_loss_history], 
                    'val_roc_auc_history': [val_roc_auc_history], 
                    'val_pr_auc_history': [val_pr_auc_history], 
                    'val_acc_history': [val_acc_history],
                    'train_loss': [train_loss_history[-1]], 
                    'train_roc_auc': [train_roc_auc_history[-1]],
                    'train_pr_auc': [train_pr_auc_history[-1]], 
                    'train_acc': [train_acc_history[-1]], 
                    'val_loss': [val_loss_history[-1]], 
                    'val_roc_auc': [val_roc_auc_history[-1]], 
                    'val_pr_auc': [val_pr_auc_history[-1]], 
                    'val_acc': [val_acc_history[-1]],
                }

                # record and experiment results and save to csv
                output_filename = "experiment_results.csv"
                pd.DataFrame(experiment_record).to_csv(output_filename, mode = 'a', header = os.path.exists(output_filename) == False, index = False)

                experiment_id += 1
                end_time = time.time()
                print("Time taken:", (end_time - start_time)/60)
                print()