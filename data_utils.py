# implements the torch datasets and data loaders for the openmic dataset
import h5py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, Sampler
from PIL import Image


def binarize_targets(targets, threshold=0.5):
    targets[targets < threshold] = 0
    targets[targets > 0] = 1
    return targets

def binary2categorical(targets):
    # Input is 20 dimensional. Output is 20x2 dimensional
    categorical = np.zeros((targets.shape[0], 2))
    categorical[targets == 1, 1] = 1
    categorical[targets == 0, 0] = 1
    return categorical

def train_val_split(full_dataset, val_ratio, aug=False):
    length = len(full_dataset)
    val_indices = np.random.choice(length, int(length*val_ratio), replace=False)
    train_indices = list(set(np.arange(length))- set(val_indices))
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    return train_dataset, val_dataset

def get_inst_datasets(npz_path, pos_weight_path, ord_path):
    data = np.load(npz_path)
    y_masks = data['Y_mask']
    full_dataset = MICDataset(npz_path, pos_weight_path, ord_path, missing=False)
    y_trues = full_dataset.Y_true
    inst_datasets = []
    for i in range(20):
        indices = np.where(y_masks[:,i] > 0)[0]
        inst_datasets.append(Subset(full_dataset, indices))
    return full_dataset, inst_datasets

class MICDataset(Dataset):
    # Pytorch dataset for OpenMIC
    def __init__(self, npz_path, missing=False):
        data = np.load(npz_path)
        self.X = data['X']/255.0
        self.Y_true = data['Y_true']
        self.Y_true = self.Y_true >= 0.5
        self.Y_mask = data['Y_mask']
        if missing:
            self.Y_true[self.Y_mask == 0] = 1
        else:
            self.Y_true[self.Y_mask == 0] = 0
        self.length = self.X.shape[0]
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        X = self.X[index]
        Y_true = binarize_targets(self.Y_true[index])
        Y = binary2categorical(Y_true)
        Y_mask = self.Y_mask[index]
        X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
        Y_true = torch.tensor(Y_true.astype(float), requires_grad=False, dtype=torch.float32)
        Y = torch.tensor(Y, requires_grad=False, dtype=torch.float32)
        Y_mask = torch.ByteTensor(Y_mask.astype(int))
        return X, Y, Y_true, Y_mask
    
    
    
class MICDataset_ResNet(Dataset):
    # Pytorch dataset for OpenMIC
    def __init__(self, npz_path, missing=False):
        data = np.load(npz_path, allow_pickle = True)
        # print("data loaded")
        self.files = data['sample_key']
        # print("keys loaded")
        self.Y_true = data['Y_true']
        self.Y_true = self.Y_true >= 0.5
        self.Y_mask = data['Y_mask']
        if missing:
            self.Y_true[self.Y_mask == 0] = 1
        else:
            self.Y_true[self.Y_mask == 0] = 0
        self.length = self.files.shape[0]
        
    def __len__(self):
        return self.length
    

    def __getitem__(self, index):
        img_name = self.files[index]
        # print(os.getcwd())
        img_path = f'/home/studio-lab-user/sagemaker-studiolab-notebooks/ESP3201-Instrument-indentification/openmic-2018/audio-gammatone/{img_name[:3]}/{img_name}.png'
        img = Image.open(img_path)
        img = img.resize((224,224))
        X = np.array(img)/255.0
        X = X[...,:3]
        X = X.transpose(2, 0, 1)
        # print(X.shape)
        Y_true = binarize_targets(self.Y_true[index])
        Y = binary2categorical(Y_true)
        Y_mask = self.Y_mask[index]
        X = torch.tensor(X, requires_grad=False, dtype=torch.float32)
        Y_true = torch.tensor(Y_true.astype(float), requires_grad=False, dtype=torch.float32)
        Y = torch.tensor(Y, requires_grad=False, dtype=torch.float32)
        Y_mask = torch.ByteTensor(Y_mask.astype(int))
        return X, Y, Y_true, Y_mask
        
def create_train_set(train_dataset, possible_inds, args):
    if args.fixed_missing:
        if args.missing_method == 'all':
            return Subset(train_dataset, possible_inds)
