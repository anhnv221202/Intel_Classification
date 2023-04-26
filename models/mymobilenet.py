import torch
import torch.nn as nn
from torchvision.models import MobileNetV2

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DSConv1(nn.Module):
    def __init__(self, ic: int, oc: int, strid: int, pad: int):
        super(DSConv1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = ic, out_channels = ic, kernel_size = 3, stride = strid, padding = pad, groups = ic),
            nn.BatchNorm2d(ic),
            nn.ReLU(inplace = True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = ic, out_channels = oc, kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        return x




class myMobileNetV1(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(myMobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace = True)
        self.features = nn.Sequential(
            DSConv1(32, 64, 1, 1),
            DSConv1(64, 128, 2, 1),
            DSConv1(128, 128, 1, 1),
            DSConv1(128, 256, 2, 1),
            DSConv1(256, 256, 1, 1),
            DSConv1(256, 512, 2, 1),
            DSConv1(512, 512, 1, 1),
            DSConv1(512, 512, 1, 1),
            DSConv1(512, 512, 1, 1),
            DSConv1(512, 512, 1, 1),
            DSConv1(512, 512, 1, 1),
            DSConv1(512, 1024, 2, 1),
            DSConv1(1024, 1024, 1, 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = 1024, out_features = num_classes)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class myIRBlock1(nn.Module):
    def __init__(self, ic: int, mul: int, oc: int, strid: int) -> None:
        super(myIRBlock1, self).__init__()
        mc = mul * ic
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, mc, 1, 1, 0),
            nn.BatchNorm2d(mc),
            nn.ReLU6(inplace = True)
        )

        self.dw1 = nn.Sequential(
            nn.Conv2d(mc, mc, 3, strid, 1, groups = mc),
            nn.BatchNorm2d(mc),
            nn.ReLU6(inplace = True),
        )

        self.project1 = nn.Sequential(
            nn.Conv2d(mc, oc, 1, 1, 0),
            nn.BatchNorm2d(oc)
        )
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.dw1(x)
        x = self.project1(x)
        return x


class myIRBlock2(nn.Module):
    def __init__(self, ioc: int, mul: int) -> None:
        super(myIRBlock2, self).__init__()
        mc = ioc * mul
        self.expand = nn.Sequential(
            nn.Conv2d(ioc, mc, 1, 1, 0),
            nn.BatchNorm2d(mc),
            nn.ReLU6(inplace = True)
        )

        self.dw = nn.Sequential(
            nn.Conv2d(mc, mc, 3, 1, 1, groups = mc),
            nn.BatchNorm2d(mc),
            nn.ReLU6(inplace = True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(mc, ioc, 1, 1, 0),
            nn.BatchNorm2d(ioc)
        )

    def forward(self, x: torch.Tensor):
        x0 = x.clone()
        x = self.expand(x)
        x = self.dw(x)
        x = self.project(x)
        x = F.relu6(x + x0)
        return x


class myMobileNetV2(nn.Module): ## currently unavailable
    def __init__(self, num_classes: int = 6):
        super(myMobileNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace = True)
        )

        self.dw1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups = 32),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace = True),
            nn.Conv2d(32, 16, 1, 1, 0),
            nn.BatchNorm2d(16),
        )

        self.blocks = nn.Sequential(
            myIRBlock1(16, 6, 24, 2),
            myIRBlock2(24, 6),
            myIRBlock1(24, 6, 32, 2),
            myIRBlock2(32, 6),
            myIRBlock2(32, 6),
            myIRBlock1(32, 6, 64, 2),
            myIRBlock2(64, 6),
            myIRBlock2(64, 6),
            myIRBlock2(64, 6),
            myIRBlock1(64, 6, 96, 1),
            myIRBlock2(96, 6),
            myIRBlock2(96, 6),
            myIRBlock1(96, 6, 160, 2),
            myIRBlock2(160, 6),
            myIRBlock2(160, 6),
            myIRBlock1(160, 6, 320, 1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace = True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1280, num_classes)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.dw1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# model = myMobileNetV2()
# print(model)
# summary(model, (3, 224, 224), device = 'cpu')