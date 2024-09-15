# Import Libraries
import os
import gc
import math
import copy
import time
import random
import glob
import timm
import cv2
import h5py

from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from PIL import Image
from io import BytesIO

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torcheval.metrics.functional import binary_auroc

from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.utils import resample

import joblib
from tqdm import tqdm
from collections import defaultdict

import albumentations as A
from albumentations.pytorch import ToTensorV2

import optuna


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# Setting Seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()


# Load Train Dataset
root_dir = "../dataset/isic-skin-cancer"
train_image_dir = f'{root_dir}/train-image/image'

def get_train_file_path(image_id):
    return f"{train_image_dir}/{image_id}.jpg"

train_images = sorted(glob.glob(f"{train_image_dir}/*.jpg"))


# Load Extra Train Dataset
train_2018_image_dir = "../dataset/isic-2018/train-image/image"
train_2019_image_dir = "../dataset/isic-2019/train-image/image"
train_2020_image_dir = "../dataset/isic-2020/train-image/image"

def get_2018_train_file_path(image_id):
    return f"{train_2018_image_dir}/{image_id}.jpg"

def get_2019_train_file_path(image_id):
    return f"{train_2019_image_dir}/{image_id}.jpg"

def get_2020_train_file_path(image_id):
    return f"{train_2020_image_dir}/{image_id}.jpg"

train_2018_images = sorted(glob.glob(f"{train_2018_image_dir}/*.jpg"))
train_2019_images = sorted(glob.glob(f"{train_2019_image_dir}/*.jpg"))
train_2020_images = sorted(glob.glob(f"{train_2020_image_dir}/*.jpg"))


# Reduce Data Imbalance
image_df = pd.read_csv(f"{root_dir}/train-metadata.csv")

print("df.shape, # of positive cases, # of patients")
print("original>", image_df.shape, image_df.target.sum(), image_df["patient_id"].unique().shape)

image_df_postive = image_df[image_df["target"] == 1].reset_index(drop=True)
image_df_negative = image_df[image_df["target"] == 0].reset_index(drop=True)

image_df = pd.concat([image_df_postive, image_df_negative.iloc[:image_df_postive.shape[0]*20, :]])
print("filtered>", image_df.shape, image_df.target.sum(), image_df["patient_id"].unique().shape)

image_df['file_path'] = image_df['isic_id'].apply(get_train_file_path)
image_df = image_df[ image_df["file_path"].isin(train_images) ].reset_index(drop=True)
image_df = image_df[['isic_id', 'target', 'patient_id', 'file_path']]
print(image_df.isnull().sum())
image_df.head()

# Reduce Data Imbalance - Extra Dataset (2018)
image_2018_df = pd.read_csv("../dataset/isic-2018/train-metadata.csv")

print("df.shape, # of positive cases, # of patients")
print("original>", image_2018_df.shape, image_2018_df.target.sum(), image_2018_df["patient_id"].unique().shape)

image_2018_df_postive = image_2018_df[image_2018_df["target"] == 1].reset_index(drop=True)
image_2018_df_negative = image_2018_df[image_2018_df["target"] == 0].reset_index(drop=True)

print(image_2018_df_postive.shape, image_2018_df_negative.shape)

image_2018_df = pd.concat([image_2018_df_postive, image_2018_df_negative.iloc[:image_2018_df_postive.shape[0]*20, :]])
print("filtered>", image_2018_df.shape, image_2018_df.target.sum(), image_2018_df["patient_id"].unique().shape)

image_2018_df['file_path'] = image_2018_df['isic_id'].apply(get_2018_train_file_path)
image_2018_df = image_2018_df[ image_2018_df["file_path"].isin(train_2018_images) ].reset_index(drop=True)
image_2018_df = image_2018_df.drop('Unnamed: 0', axis=1)
print(image_2018_df.columns)
print(image_2018_df.isnull().sum())
image_2018_df.head()

# Reduce Data Imbalance - Extra Dataset (2019)
image_2019_df = pd.read_csv("../dataset/isic-2019/train-metadata.csv")

print("df.shape, # of positive cases, # of patients")
print("original>", image_2019_df.shape, image_2019_df.target.sum(), image_2019_df["patient_id"].unique().shape)

image_2019_df_postive = image_2019_df[image_2019_df["target"] == 1].reset_index(drop=True)
image_2019_df_negative = image_2019_df[image_2019_df["target"] == 0].reset_index(drop=True)

print(image_2019_df_postive.shape, image_2019_df_negative.shape)

image_2019_df = pd.concat([image_2019_df_postive, image_2019_df_negative.iloc[:image_2019_df_postive.shape[0]*20, :]])
print("filtered>", image_2019_df.shape, image_2019_df.target.sum(), image_2019_df["patient_id"].unique().shape)

image_2019_df['file_path'] = image_2019_df['isic_id'].apply(get_2019_train_file_path)
image_2019_df = image_2019_df[ image_2019_df["file_path"].isin(train_2019_images) ].reset_index(drop=True)
image_2019_df = image_2019_df.drop('Unnamed: 0', axis=1)
print(image_2019_df.columns)
print(image_2019_df.isnull().sum())
image_2019_df.head()

# Reduce Data Imbalance - Extra Dataset (2020)
image_2020_df = pd.read_csv("../dataset/isic-2020/train-metadata.csv")

print("df.shape, # of positive cases, # of patients")
print("original>", image_2020_df.shape, image_2020_df.target.sum(), image_2020_df["patient_id"].unique().shape)

image_2020_df_postive = image_2020_df[image_2020_df["target"] == 1].reset_index(drop=True)
image_2020_df_negative = image_2020_df[image_2020_df["target"] == 0].reset_index(drop=True)

print(image_2020_df_postive.shape, image_2020_df_negative.shape)

image_2020_df = pd.concat([image_2020_df_postive, image_2020_df_negative.iloc[:image_2020_df_postive.shape[0]*20, :]])
print("filtered>", image_2020_df.shape, image_2020_df.target.sum(), image_2020_df["patient_id"].unique().shape)

image_2020_df['file_path'] = image_2020_df['isic_id'].apply(get_2020_train_file_path)
image_2020_df = image_2020_df[ image_2020_df["file_path"].isin(train_2020_images) ].reset_index(drop=True)
image_2020_df = image_2020_df.drop('Unnamed: 0', axis=1)
print(image_2020_df.columns)
print(image_2020_df.isnull().sum())
image_2020_df.head()

combined_df = pd.concat([image_df, image_2018_df, image_2019_df, image_2020_df], axis=0, ignore_index=True)
print(combined_df.shape)
combined_df

print(combined_df.isnull().sum())

t_max_value = combined_df.shape[0] * (4) * 10 // 32 // 5
print(f"t_max_value: {t_max_value}")


# K-Fold
skf = StratifiedGroupKFold(n_splits=5)

for fold, ( _, val_) in enumerate(skf.split(combined_df, combined_df.target, combined_df.patient_id)):
      combined_df.loc[val_ , "kfold"] = int(fold)


# Make DataLoader
class ISICDataset_for_Train(Dataset):
    def __init__(self, df, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df_positive) * 3
    
    def __getitem__(self, index):
        if random.random() >= 0.76:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]
        
        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }

    
class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return {
            'image': img,
            'target': target
        }
    

# Augmentation
data_transforms = {
    "train": A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(224, 224),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


# GeM Pooling Layer
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1./p)
    

# ISIC Model (DINO)
def normalize_image(image_tensor):
    min_val = image_tensor.min()
    max_val = image_tensor.max()
    normalized_tensor = (image_tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class ISICDINOModel(nn.Module):
    def __init__(self, num_classes=1):
        super(ISICDINOModel, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('./model')
        self.model = AutoModel.from_pretrained('./model')
        
        self.in_features = self.model.config.hidden_size
        
        self.gem_pooling = GeM()
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = normalize_image(x)

        inputs = self.processor(images=x, return_tensors="pt").to(x.device)

        outputs = self.model(**inputs)
        features = outputs.last_hidden_state

        features = features.mean(dim=1).unsqueeze(-1)
        pooled_features = self.gem_pooling(features).squeeze(-1)

        pooled_features = self.dropout(pooled_features)
        logits = self.fc(pooled_features)
        output = self.sigmoid(logits)

        return output.squeeze(-1)
    
model = ISICDINOModel()
model.to(device)


test_input = torch.randn(32, 3, 224, 224).cuda()
output = model(test_input)
print(f"Output shape: {output.shape}")
print(output)


# Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

criterion = FocalLoss()



# Training & Validation
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc  = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss = loss / 1
            
        loss.backward()
    
        if (step + 1) % 1 == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss, epoch_auroc


@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        running_loss += (loss.item() * batch_size)
        running_auroc  += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    
    return epoch_loss, epoch_auroc

def run_training(model, optimizer, scheduler, device, num_epochs):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_auroc = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)
        
        val_epoch_loss, val_epoch_auroc = valid_one_epoch(model, valid_loader, device=device, 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train AUROC'].append(train_epoch_auroc)
        history['Valid AUROC'].append(val_epoch_auroc)
        history['lr'].append( scheduler.get_lr()[0] )
        
        # deep copy the model
        if best_epoch_auroc <= val_epoch_auroc:
            print(f"{b_}Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
            best_epoch_auroc = val_epoch_auroc
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "vit_best_weight.bin".format(val_epoch_auroc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history



# Schedular
def fetch_scheduler(optimizer, name):
    if name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=t_max_value, 
                                                   eta_min=5e-7)
    elif name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=100, 
                                                             eta_min=1e-6)
    elif name == None:
        return None
        
    return scheduler


def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = ISICDataset_for_Train(df_train, transforms=data_transforms["train"])
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=32, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader


train_loader, valid_loader = prepare_loaders(combined_df, fold=0)

optimizer = optim.AdamW([
    {'params': model.model.parameters(), 'lr': 1e-5},  # Transformer model parameters
    {'params': model.fc.parameters(), 'lr': 1e-4}  # Classification head parameters
], weight_decay=0.01)

scheduler = fetch_scheduler(optimizer, 'CosineAnnealingLR')

model, history = run_training(model, optimizer, scheduler,
                              device=device,
                              num_epochs=50)



# Logging
history = pd.DataFrame.from_dict(history)
history.to_csv("DINO_history.csv", index=False)