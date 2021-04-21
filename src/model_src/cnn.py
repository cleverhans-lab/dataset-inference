import torch
import torch.nn as nn
import ipdb
import torch.multiprocessing as _mp
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, layer_num = -1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding = 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding = 2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024,10)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer_num = layer_num
        self.outputs = []

    def forward(self,x):
        self.outputs = []
        x = self.conv1(x)
        self.outputs.append(x.cpu().detach())
        x = self.maxpool(nn.ReLU()(x))
        x = self.conv2(x)
        self.outputs.append(x.cpu().detach())
        x = self.maxpool(nn.ReLU()(x))
        x = x.view(x.shape[0], -1)  
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        self.outputs.append(x.cpu().detach())
        return x


    def forward_features(self,x):
        features = x.clone()
        x = nn.ReLU()(self.conv1(x))
        x = self.maxpool(x); features = x.clone() if self.layer_num == 1 else features
        x = nn.ReLU()(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1); features = x.clone() if self.layer_num == 2 else features
        x = nn.ReLU()(self.fc1(x)); features = x.clone() if self.layer_num == 3 else features
        x = self.fc2(x)
        return x, features

    def features(self,x):
        x = nn.ReLU()(self.conv1(x)) if self.layer_num > 0 else x
        x = self.maxpool(x) if self.layer_num > 0 else x
        x = nn.ReLU()(self.conv2(x)) if self.layer_num > 1 else x
        x = self.maxpool(x) if self.layer_num > 2 else x
        x = x.view(x.shape[0], -1) if self.layer_num > 2 else x
        x = nn.ReLU()(self.fc1(x)) if self.layer_num > 2 else x
        x = self.fc2(x) if self.layer_num > 2 else x
        return x

class ClassifierCNN(nn.Module):
    def __init__(self, num_attacks=3, layer_num = 1, fac = 16):
        super(ClassifierCNN, self).__init__()
        assert (layer_num in [0,1,2,3])
        self.in_planes = [1, 32, 7*7*64, 1024]
        self.out_planes = [32, 64, 1024, num_attacks]

        self.in_planes[layer_num] = self.in_planes[layer_num]*3
        self.out_planes[layer_num - 1] = self.out_planes[layer_num - 1]*3 if layer_num != 0 else self.out_planes[layer_num - 1]

        assert (fac==1 or layer_num!=3)
        if layer_num != 3:
            self.in_planes[layer_num + 1] = self.in_planes[layer_num + 1]*fac
            self.out_planes[layer_num] = self.out_planes[layer_num]*fac


        self.layer_num = layer_num

        self.conv1 = nn.Conv2d(self.in_planes[0], self.out_planes[0], 5, padding = 2) if self.layer_num == 0 else None
        self.conv2 = nn.Conv2d(self.in_planes[1], self.out_planes[1], 5, padding = 2) if self.layer_num <= 1 else None
        self.fc1 = nn.Linear(self.in_planes[2], self.out_planes[2]) if self.layer_num <= 2 else None
        self.fc2 = nn.Linear(self.in_planes[3], self.out_planes[3])
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self,x):
        x = nn.ReLU()(self.conv1(x)) if self.layer_num == 0 else x
        x = self.maxpool(x) if self.layer_num == 0 else x
        x = nn.ReLU()(self.conv2(x)) if self.layer_num <= 1 else x
        x = self.maxpool(x) if self.layer_num <= 1 else x
        x = x.view(x.shape[0], -1)  if self.layer_num <= 1 else x
        x = nn.ReLU()(self.fc1(x)) if self.layer_num <= 2 else x
        x = self.fc2(x) if self.layer_num <= 2 else x
        return x

class VanillaClassifierCNN(nn.Module):
    def __init__(self, num_attacks=3, layer_num = 1, fac = 16):
        super(VanillaClassifierCNN, self).__init__()
        print("Num_Attacks = ", num_attacks)
        assert (layer_num in [0,1,2,3])
        self.in_planes = [1, 32, 7*7*64, 1024]
        self.out_planes = [32, 64, 1024, num_attacks]

        # self.in_planes[layer_num] = self.in_planes[layer_num]*3
        # self.out_planes[layer_num - 1] = self.out_planes[layer_num - 1]*3 if layer_num != 0 else self.out_planes[layer_num - 1]

        # assert (fac==1 or layer_num!=3)
        # if layer_num != 3:
        #     self.in_planes[layer_num + 1] = self.in_planes[layer_num + 1]*fac
        #     self.out_planes[layer_num] = self.out_planes[layer_num]*fac
        # assert (fac == 1 and layer_num == 0 )

        in_planes = self.out_planes[layer_num-1] if layer_num > 0 else 1
        self.in_planes = [in_planes, 32, 3*3*64, 1024] if layer_num > 0 else [in_planes, 32, 7*7*64, 1024]
        layer_num = 0
        self.out_planes = [32, 64, 1024, num_attacks]

        self.layer_num = layer_num

        self.conv1 = nn.Conv2d(self.in_planes[0], self.out_planes[0], 5, padding = 2) if self.layer_num == 0 else None
        self.conv2 = nn.Conv2d(self.in_planes[1], self.out_planes[1], 5, padding = 2) if self.layer_num <= 1 else None
        self.fc1 = nn.Linear(self.in_planes[2], self.out_planes[2]) if self.layer_num <= 2 else None
        self.fc2 = nn.Linear(self.in_planes[3], self.out_planes[3])
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self,x):
        x = nn.ReLU()(self.conv1(x)) if self.layer_num == 0 else x
        x = self.maxpool(x) if self.layer_num == 0 else x
        x = nn.ReLU()(self.conv2(x)) if self.layer_num <= 1 else x
        x = self.maxpool(x) if self.layer_num <= 1 else x
        x = x.view(x.shape[0], -1)  if self.layer_num <= 1 else x
        x = nn.ReLU()(self.fc1(x)) if self.layer_num <= 2 else x
        x = self.fc2(x) if self.layer_num <= 2 else x
        return x


