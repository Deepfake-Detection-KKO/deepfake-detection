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
NUM_WORKERS = 8

# Enable TensorFloat32 for better performance on compatible GPUs
torch.set_float32_matmul_precision('high')

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
IMAGE_DIR_PATH = 'Deepfake-Eval-2024/image-data-rescaled'

# Device
device = torch.accelerator.current_accelerator()
print(f'Using {device} accelerator \n')

# Save Model Weights?
save_model_weights = False

# Loss function
loss_fn = nn.CrossEntropyLoss(reduction = 'sum')

# Hyperparameters TODO update
model_types = ['ConvNeXt-base-pretrained', 'ViT-b32-pretrained', 'ResNet-50-pretrained']
freeze_layers = [False, True]
dropout_rates = [0, 0.2]
l2_penalties = [0, 0.0001]
optimizer_classes = [torch.optim.Adam]
learning_rates = [1e-3]
epochs_list = [10]
lr_scheduler_types = ['StepLR', 'CosineAnnealingWarmRestarts']

experiment_id = 1

# Iterate through hyperparameters
for model_type in model_types:
    if "ViT" in model_type:
        transform_type = 'ViT'
    elif "ResNet" in model_type:
        transform_type = 'ResNet'
    elif "ConvNeXt" in model_type:
        transform_type  = 'ConvNeXt'
    
    # Load data
    train_data = DeepFakeDataset("image-metadata-train.csv", IMAGE_DIR_PATH, transform_type, is_train = True)
    train_data_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers=NUM_WORKERS)

    val_data = DeepFakeDataset("image-metadata-val.csv", IMAGE_DIR_PATH, transform_type, is_train = False)
    val_data_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = False, num_workers=NUM_WORKERS)

    for freeze_layer in freeze_layers:
        for dropout_rate in dropout_rates:
            for l2_penalty in l2_penalties:
                for optimizer_class in optimizer_classes:
                    for epochs in epochs_list:
                        for learning_rate in learning_rates:
                            for lr_scheduler_type in lr_scheduler_types:
                                start_time = time.time()

                                # Create model
                                model = MyModel(
                                    model_type=model_type,
                                    device=device,
                                    dropout_rate=dropout_rate,
                                    freeze_layers=freeze_layer
                                )
                                model = torch.compile(model)

                                # Optimizer and learning rate schedule
                                optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
                                lr_scheduler = None
                                if lr_scheduler_type == 'CosineAnnealingWarmRestarts':
                                    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                        optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
                                    )
                                elif lr_scheduler_type == 'StepLR':
                                    lr_scheduler = torch.optim.lr_scheduler.StepLR(
                                        optimizer, step_size=2, gamma=0.5
                                    )
                                
                                # Train the model
                                total_experiments = len(model_types) * len(freeze_layers) * len(dropout_rates) * len(l2_penalties) * len(optimizer_classes) * len(learning_rates) * len(epochs_list) * len(lr_scheduler_types)
                                optimizer_name = optimizer_class.__name__
                                print(f'Exp {experiment_id} of {total_experiments}: Model={model_type}, Optim={optimizer_name}, LR={learning_rate}, L2={l2_penalty}, Dropout={dropout_rate}, Scheduler={lr_scheduler_type}, Freeze={freeze_layer}')
                                train_loss_history, train_roc_auc_history, train_pr_auc_history, train_acc_history, val_loss_history, val_roc_auc_history, val_pr_auc_history, val_acc_history = train(
                                    epochs,
                                    train_data_loader,
                                    val_data_loader,
                                    model,
                                    loss_fn,
                                    optimizer,
                                    device,
                                    lr_scheduler=lr_scheduler)
                                
                                experiment_record = {
                                    'experiment_id': experiment_id,
                                    'model': model_type,
                                    'freeze_layers': freeze_layer,
                                    'dropout_rate': dropout_rate,
                                    'l2-penalty': l2_penalty,
                                    'optimizer': optimizer_name,
                                    'epochs': epochs,
                                    'learning_rate': learning_rate,
                                    'lr_scheduler_type': lr_scheduler_type,
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
                                if save_model_weights:
                                    torch.save(model.state_dict(), f"experiment_{experiment_id}.pth")