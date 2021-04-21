import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ipdb
import random
import torchvision
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from attacks import *
import pandas as pd
import copy, sys

class PseudoDataset(torch.utils.data.Dataset):
    
    def __init__(self, x, y, transform = None):
        self.x_data = x
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]


    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return (x_data_index, self.y_data[index])

    def __len__(self):
        return self.len

class AFADDatasetAge(Dataset):
    """Custom Dataset for loading AFAD face images"""

    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img = pil_loader(os.path.join(self.img_dir,
                                      self.img_paths[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]



def get_dataloaders(dataset, batch_size, pseudo_labels = False, normalize = False, train_shuffle = True, concat = False, concat_factor = 1.0):
    if dataset in ["CIFAR10", "CIFAR100"]:
        data_source = datasets.CIFAR10 if dataset == "CIFAR10" else datasets.CIFAR100
        tr_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) if normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), tr_normalize,
                                    transforms.Lambda(lambda x: x.float())])

        transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])
        if not train_shuffle:
            print("No Transform")
            transform_train = transform_test
        cifar_train = data_source("../../data", train=True, download=True, transform=transform_train)
        cifar_test = data_source("../../data", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(cifar_test, batch_size = batch_size, shuffle=False)

        if pseudo_labels:
            transform_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])
            
            import pickle, os
            aux_data_filename = "ti_500K_pseudo_labeled.pickle"
            aux_path = os.path.join("../data", aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']

            cifar_train = PseudoDataset(aux_data, aux_targets, transform = transform_train)
            train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=train_shuffle)
    
    if dataset == "MNIST":
        tr_normalize = transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([transforms.ToTensor(), tr_normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize]) #Change
        mnist_train = datasets.MNIST("../../data", train=True, download=True, transform=transform_train)
        mnist_test = datasets.MNIST("../../data", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle=train_shuffle)
        test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle=False)

    
    elif dataset != "AFAD":
        func = {"SVHN":datasets.SVHN, "CIFAR10":datasets.CIFAR10, "CIFAR100":datasets.CIFAR100, "MNIST":datasets.MNIST, "ImageNet":datasets.ImageNet}
        norm_mean = {"SVHN":(0.438, 0.444, 0.473), "CIFAR10":(0.4914, 0.4822, 0.4465), "CIFAR100":(0.4914, 0.4822, 0.4465), "MNIST":(0.1307,), "ImageNet":(0.485, 0.456, 0.406)}
        norm_std = {"SVHN":(0.198, 0.201, 0.197), "CIFAR10":(0.2023, 0.1994, 0.2010), "CIFAR100":(0.2023, 0.1994, 0.2010), "MNIST": (0.3081,), "ImageNet":(0.229, 0.224, 0.225)}


        tr_normalize = transforms.Normalize(norm_mean[dataset], norm_std[dataset]) if normalize else transforms.Lambda(lambda x: x)
        transform_train = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), tr_normalize,
                                        transforms.Lambda(lambda x: x.float())])
        transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])

        data_source = func[dataset]
        if not train_shuffle:
            print("No Transform")
            transform_train = transform_test
        
        if dataset == "ImageNet":
            transform = transforms.Compose([transforms.Resize(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.ToTensor(),
                                            tr_normalize])
            train_path = '/scratch/ssd001/datasets/imagenet/train'
            test_path = '/scratch/ssd001/datasets/imagenet/val'

            d_train = torchvision.datasets.ImageFolder(train_path, transform=transform)
            d_test = torchvision.datasets.ImageFolder(test_path, transform=transform)
            # d_test = data_source("/scratch/ssd001/datasets/imagenet", split='val', download=False, transform=transform_test)
            # d_train = data_source("/scratch/ssd001/datasets/imagenet", split='train', download=False, transform=transform_train)
        else:
            try:
                d_train = data_source("../data", train=True, download=True, transform=transform_train)
                d_test = data_source("../data", train=False, download=True, transform=transform_test)
            except:
                if concat :
                    d_train = data_source("../data", split='train', download=True, transform=transform_train)
                    d_extra = data_source("../data", split='extra', download=True, transform=transform_test)
                    train_len = d_train.data.shape[0]
                    new_len = int(train_len*concat_factor)
                    d_train.data = d_train.data[:new_len]
                    d_train.labels = d_train.labels[:new_len]
                    d_extra.data = d_extra.data[:50000]
                    d_extra.labels = d_extra.labels[:50000]
                    d_train = torch.utils.data.ConcatDataset([d_train,d_extra])
                
                else:
                    d_train = data_source("../data", split='train' if not pseudo_labels else 'extra', download=True, transform=transform_train)
                    d_train.data = d_train.data[:50000]
                    d_train.labels = d_train.labels[:50000]

                d_test = data_source("../data", split='test', download=True, transform=transform_test)
                
        train_loader = DataLoader(d_train, batch_size = batch_size, shuffle=True)
        test_loader = DataLoader(d_test, batch_size = batch_size, shuffle=False)
        if pseudo_labels and dataset != "SVHN":
            transform_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])
            import pickle, os
            aux_data_filename = "ti_500K_pseudo_labeled.pickle"
            aux_path = os.path.join("../data", aux_data_filename)
            print("Loading data from %s" % aux_path)
            with open(aux_path, 'rb') as f:
                aux = pickle.load(f)
            aux_data = aux['data']
            aux_targets = aux['extrapolated_targets']

            cifar_train = PseudoDataset(aux_data, aux_targets, transform = transform_train)
            train_loader = DataLoader(cifar_train, batch_size = batch_size, shuffle=train_shuffle)

    else:
        TRAIN_CSV_PATH = '/home/users/myaghini/AFAD_saved_model/afad_train.csv'
        TEST_CSV_PATH = '/home/users/myaghini/AFAD_saved_model/afad_test.csv'
        IMAGE_PATH = '/home/users/myaghini/AFAD-Full'
        NUM_WORKERS = 3
        train_transform = transforms.Compose([transforms.Resize((128, 128)),transforms.RandomCrop((120, 120)),transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((128, 128)),transforms.CenterCrop((120, 120)),transforms.ToTensor()])

        train_dataset = AFADDatasetAge(csv_path=TRAIN_CSV_PATH, img_dir=IMAGE_PATH,transform=train_transform)
        test_dataset = AFADDatasetAge(csv_path=TEST_CSV_PATH,img_dir=IMAGE_PATH, transform=test_transform)


        train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=NUM_WORKERS)
        test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=NUM_WORKERS)

        
    return train_loader, test_loader



def norms(Z):
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def load(model, model_name):
    try:
        model.load_state_dict(torch.load(f"{model_name}.pt"))
    except:
        dictionary = torch.load(f"{model_name}.pt")['state_dict']
        new_dict = {}
        for key in dictionary.keys():
            new_key = key[7:]
            if new_key.split(".")[0] == "sub_block1":
                continue
            new_dict[new_key] = dictionary[key]
        model.load_state_dict(new_dict)
    return model
