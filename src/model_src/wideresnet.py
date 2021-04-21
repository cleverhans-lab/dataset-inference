import torch
import torch.nn as nn
import math
import torch.nn.functional as F
## Use the same mean for Tiny images, CIFAR10 and CIFAR100 since they are from nearly similar distributions

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'svhn':(0.438, 0.444, 0.473)
}

std = {
    # cifar10_std = (0.2023, 0.1994, 0.2010) #Some repositories use this value
    'cifar10': (0.2471, 0.2435, 0.2616),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'svhn':(0.198, 0.201, 0.197)
}


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class IndividualBlock1(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, subsample_input=True, increase_filters=True):
        super(IndividualBlock1, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropRate)
        self.batch_norm1 = nn.BatchNorm2d(in_planes)
        self.batch_norm2 = nn.BatchNorm2d(out_planes)

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.subsample_input = subsample_input
        self.increase_filters = increase_filters
        if subsample_input:
            self.conv_inp = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2, padding=0, bias=False)
        elif increase_filters:
            self.conv_inp = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):

        if self.subsample_input or self.increase_filters:
            x = self.batch_norm1(x)
            x = self.activation(x)
            x1 = self.conv1(x)
        else:
            x1 = self.batch_norm1(x)
            x1 = self.activation(x1)
            x1 = self.conv1(x1)
        
        
        x1 = self.batch_norm2(x1)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)
        x1 = self.conv2(x1)

        if self.subsample_input or self.increase_filters:
            return self.conv_inp(x) + x1
        else:
            return x + x1


class IndividualBlockN(nn.Module):

    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(IndividualBlockN, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropRate)
        self.batch_norm1 = nn.BatchNorm2d(in_planes)
        self.batch_norm2 = nn.BatchNorm2d(out_planes)

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = self.activation(x1)
        x1 = self.conv1(x1)
        x1 = self.batch_norm2(x1)
        x1 = self.activation(x1)
        x1 = self.dropout(x1)
        x1 = self.conv2(x1)

        return x1 + x


class Nblock(nn.Module):

    def __init__(self, N, in_planes, out_planes, stride, dropRate=0.0, subsample_input=True, increase_filters=True):
        super(Nblock, self).__init__()

        layers = []
        for i in range(N):
            if i == 0:
                layers.append(IndividualBlock1(in_planes, out_planes, stride, dropRate, subsample_input, increase_filters))
            else:
                layers.append(IndividualBlockN(out_planes, out_planes, stride=1, dropRate=dropRate))

        self.nblockLayer = nn.Sequential(*layers)

    def forward(self, x):
        return self.nblockLayer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, n_classes=10, in_planes=3, out_planes=16, strides = [1, 1, 2, 2], dropRate = 0.0, normalize = True):
        super(WideResNet, self).__init__()
        k = widen_factor
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=strides[0], padding=1, bias=False)
        self.normalize = normalize

        filters = [16 * k, 32 * k, 64 * k]
        self.out_filters = filters[-1]
        N = (depth - 4) // 6
        increase_filters = k > 1
        self.block1 = Nblock(N, in_planes=out_planes, out_planes=filters[0], stride=strides[1], dropRate = dropRate, subsample_input=False, increase_filters=increase_filters)
        self.block2 = Nblock(N, in_planes=filters[0], out_planes=filters[1], stride=strides[2], dropRate = dropRate)
        self.block3 = Nblock(N, in_planes=filters[1], out_planes=filters[2], stride=strides[3], dropRate = dropRate)

        self.batch_norm = nn.BatchNorm2d(filters[-1])
        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(filters[-1], n_classes)
        dataset = "CIFAR10" if n_classes == 10 else "CIFAR100"
        self.mu = torch.tensor(mean[dataset.lower()]).view(3,1,1)
        self.std = torch.tensor(std[dataset.lower()]).view(3,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        #Normalize 
        if self.normalize: 
            mu = self.mu.to(x.device)
            std = self.std.to(x.device)
            x = (x-mu)/std
        x = self.conv1(x)
        attention1 = self.block1(x)
        attention2 = self.block2(attention1)
        attention3 = self.block3(attention2)
        out = self.batch_norm(attention3)
        out = self.activation(out)
        out = self.avg_pool(out)
        out = out.view(-1, self.out_filters)

        return self.fc(out)#, attention1, attention2, attention3
