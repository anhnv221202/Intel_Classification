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


class myIRBlock(nn.Module):
    def __init__(self, ic: int, oc: int, multi: int, num_block: int, strid: int):
        super(myIRBlock, self).__init__()
        mc = ic * multi
        self.block1 = nn.Sequential(
            nn.Conv2d(ic, mc, 1, 1, 0),
            nn.BatchNorm2d(mc),
            nn.ReLU6(inplace = True)
        )
        if num_block == 2:
            self.block2 = nn.Sequential()
        else:
            self.block2 = nn.Sequential(
                nn.Conv2d(mc, mc, 3, strid, 1),
                nn.BatchNorm2d(mc),
                nn.ReLU6(inplace = True)
            )
        self.block3 = nn.Sequential(
            nn.Conv2d(mc, oc, 1, 1, 0),
            nn.BatchNorm2d(oc),

        )


        self.shortcut = nn.Sequential()
        if strid == 2 or ic != oc:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ic, oc, 1, strid, 0),
                nn.BatchNorm2d(oc),
                nn.ReLU6(inplace = True)
            )
    def forward(self, x: torch.Tensor):
        x0 = x.clone()

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)


        x0 = self.shortcut(x0)
        x = x + x0
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

class myMobileNetV2(nn.Module): ## currently unavailable
    def __init__(self, num_classes: int = 6):
        super(myMobileNetV2, self).__init__()
        pass
    def forward(self, x: torch.Tensor):
        return x

# class BIMobileNetV2(MobileNetV2):
#     def __init__(self, num_classes: int = 6, width_mult: float = 1, inverted_residual_setting = None, round_nearest: int = 8, block = None, norm_layer= None, dropout: float = 0.2) -> None:
#         super().__init__(num_classes, width_mult, inverted_residual_setting, round_nearest, block, norm_layer, dropout)

# model = myMobileNetV1()
# print(model)
# summary(model, (3, 224, 224), device = 'cpu')