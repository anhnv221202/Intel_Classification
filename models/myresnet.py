import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import torch.nn.functional as F

class myresblock(nn.Module):
    def __init__(self, ic: int, oc: int):
        super(myresblock, self).__init__()

        stride = 1 if ic == oc else 2
        self.conv1 = nn.Conv2d(in_channels = ic, out_channels = oc, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(oc)
        self.relu1 = nn.ReLU(inplace = True)

        self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(oc)
        self.relu2 = nn.ReLU(inplace = True)


        self.shortcut = nn.Sequential()
        if ic != oc:
            self.shortcut.append(nn.Conv2d(ic, oc, 1, 2, 0))
            self.shortcut.append(nn.BatchNorm2d(oc))
            self.shortcut.append(nn.ReLU(inplace = True))

    def forward(self, x: torch.Tensor):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x + self.shortcut(x0)
        x = F.relu(x)
        return x
  
class mybottleneck(nn.Module):
    def __init__(self, ic: int, mc:int, oc: int):
        super(mybottleneck, self).__init__()
        strid = 1 if ic == oc else 2
        self.conv1 = nn.Conv2d(ic, mc, 1, strid, 0)
        self.bn1 = nn.BatchNorm2d(mc)

        self.conv2 = nn.Conv2d(mc, mc, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(mc)

        self.conv3 = nn.Conv2d(mc, oc, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(oc)

        self.shortcut = nn.Sequential()

        if strid == 2:
            self.shortcut.append(nn.Conv2d(ic, oc, 1, 2, 0))
            self.shortcut.append(nn.BatchNorm2d(oc))
            self.shortcut.append(nn.ReLU(inplace = True))
    def forward(self, x: torch.Tensor):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x0 = self.shortcut(x0)

        x = x + x0
        x = F.relu(x)
        return x

class myresnet34(nn.Module):
    def __init__(self, num_classes = 6):
        super(myresnet34, self).__init__()
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU(inplace = True)
        self.layer1 = nn.Sequential(
            myresblock(64, 64),
            myresblock(64, 64),
            myresblock(64, 64),
        )
        self.layer2 = nn.Sequential(
            myresblock(64, 128),
            myresblock(128, 128),
            myresblock(128, 128),
            myresblock(128, 128),


        )
        self.layer3 = nn.Sequential(
            myresblock(128, 256),
            myresblock(256, 256),
            myresblock(256, 256),
            myresblock(256, 256),
            myresblock(256, 256),
            myresblock(256, 256),

        )
        self.layer4 = nn.Sequential(
            myresblock(256, 512),
            myresblock(512, 512),
            myresblock(512, 512),

        )

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class myresnet18(nn.Module):
    def __init__(self, num_classes = 6):
        super(myresnet18, self).__init__()
        self.conv1 = nn.Conv2d(3,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.relu1 = nn.ReLU(inplace = True)
        self.layer1 = nn.Sequential(
            myresblock(64, 64),
            myresblock(64, 64),
        )
        self.layer2 = nn.Sequential(
            myresblock(64, 128),
            myresblock(128, 128),
        )
        self.layer3 = nn.Sequential(
            myresblock(128, 256),
            myresblock(256, 256),
        )
        self.layer4 = nn.Sequential(
            myresblock(256, 512),
            myresblock(512, 512)
        )

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class myresnet50(nn.Module):
    def __init__(self, num_classes = 6):
        super(myresnet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = nn.Sequential(
            mybottleneck(64, 64, 256),
            mybottleneck(256, 64, 256),
            mybottleneck(256, 64, 256)
        )

        self.layer2 = nn.Sequential(
            mybottleneck(256, 128, 512),
            mybottleneck(512, 128, 512),
            mybottleneck(512, 128, 512),
            mybottleneck(512, 128, 512),
        )

        self.layer3 = nn.Sequential(
            mybottleneck(512, 256, 1024),
            mybottleneck(1024, 256, 1024),
            mybottleneck(1024, 256, 1024),
            mybottleneck(1024, 256, 1024),
            mybottleneck(1024, 256, 1024),
            mybottleneck(1024, 256, 1024),
        )

        self.layer4 = nn.Sequential(
            mybottleneck(1024, 512, 2048),
            mybottleneck(2048, 512, 2048),
            mybottleneck(2048, 512, 2048),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        self.relu1 = nn.ReLU(inplace = True)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.relu1(x)

        return x
    
class myresnet101(nn.Module):
    def __init__(self, num_classes = 6):
        super(myresnet101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2)

        self.layer1 = nn.Sequential(
            mybottleneck(64, 64, 256),
            mybottleneck(256, 64, 256),
            mybottleneck(256, 64, 256)
        )

        self.layer2 = nn.Sequential(
            mybottleneck(256, 128, 512),
            mybottleneck(512, 128, 512),
            mybottleneck(512, 128, 512),
            mybottleneck(512, 128, 512),
        )

        self.layer3 = nn.Sequential(
            mybottleneck(512, 256, 1024),
        )

        for i in range(22):
            self.layer3.append(mybottleneck(1024, 256, 1024))

        self.layer4 = nn.Sequential(
            mybottleneck(1024, 512, 2048),
            mybottleneck(2048, 512, 2048),
            mybottleneck(2048, 512, 2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        self.relu1 = nn.ReLU(inplace = True)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.relu1(x)

        return x
    
class myresnet152(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(myresnet152, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2)

        self.layer1 = nn.Sequential(
            mybottleneck(64, 64, 256),
            mybottleneck(256, 64, 256),
            mybottleneck(256, 64, 256)
        )

        self.layer2 = nn.Sequential(
            mybottleneck(256, 128, 512),
        )

        for i in range(7):
            self.layer2.append(mybottleneck(512, 128, 512))

        self.layer3 = nn.Sequential(
            mybottleneck(512, 256, 1024),
        )

        for i in range(35):
            self.layer3.append(mybottleneck(1024, 256, 1024))

        self.layer4 = nn.Sequential(
            mybottleneck(1024, 512, 2048),
            mybottleneck(2048, 512, 2048),
            mybottleneck(2048, 512, 2048),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, num_classes)
        self.relu1 = nn.ReLU(inplace = True)
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.relu1(x)

        return x