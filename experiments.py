import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights, resnet18, ResNet18_Weights

from torch.utils.data import Dataset, DataLoader
from deepfake_utils.datasets import DeepFakeDataset
from deepfake_utils.train import train

image_dir_path = 'Deepfake-Eval-2024/image-data'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
learning_rate = 1e-3
batch_size = 32
epochs = 3
loss_fn = nn.CrossEntropyLoss(reduction = 'sum')

###############
# experiment 1
###############
# modify ConvNeXtclassifier head for deepfake detection
experiment_id = 1
# model = convnext_base(weights = ConvNeXt_Base_Weights.DEFAULT)
# classifier_head_layers = [layer for layer in model.classifier]
# classifier_head_layers[2] = nn.Linear(in_features=1024, out_features=2, bias = True)
# model.classifier = nn.Sequential(*classifier_head_layers)

model = resnet18(weights = ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(in_features = 512, out_features=2, bias = True)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train_data = DeepFakeDataset("image-metadata-debug.csv", image_dir_path, 'ResNet', is_train = True)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

val_data = DeepFakeDataset("image-metadata-debug.csv", image_dir_path, 'ResNet', is_train = False)
val_data_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)

train_loss_history, train_roc_auc_history, train_pr_auc_history, train_acc_history, val_loss_history, val_roc_auc_history, val_pr_auc_history, val_acc_history = train(epochs, train_data_loader, val_data_loader, model, loss_fn, optimizer, device)
experiment_record = {
    'experiment_id': experiment_id,
    'model': 'convnext',
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
with open(f"experiment_{experiment_id}.pth", "wb") as file:
    torch.save(model.state_dict(), file)

###############
# experiment 2
###############
# modify ResNet classifier head for deepfake detection
experiment_id = 2
model = resnet18(weights = ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(in_features = 512, out_features=2, bias = True)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train_data = DeepFakeDataset("image-metadata-debug.csv", image_dir_path, 'ResNet', is_train = True)
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

val_data = DeepFakeDataset("image-metadata-debug.csv", image_dir_path, 'ResNet', is_train = False)
val_data_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False)

train_loss_history, train_roc_auc_history, train_pr_auc_history, train_acc_history, val_loss_history, val_roc_auc_history, val_pr_auc_history, val_acc_history = train(epochs, train_data_loader, val_data_loader, model, loss_fn, optimizer, device)
experiment_record = {
    'experiment_id': experiment_id,
    'model': 'resnet',
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
output_filename = "experiment_results.csv"
pd.DataFrame(experiment_record).to_csv(output_filename, mode = 'a', header = os.path.exists(output_filename) == False, index = False)
torch.save(model.state_dict(), f"experiment_{experiment_id}.pth")