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

## block1 is image b1, block 2 is image b2: https://images.viblo.asia/a536248d-cb35-4be6-adc3-20eeb70684bd.PNG

class myIRBlock2(nn.Module):
    def __init__(self, ic: int, mul: int, oc: int, strid: int) -> None:
        super(myIRBlock2, self).__init__()
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


class myIRBlock1(nn.Module):
    def __init__(self, ioc: int, mul: int) -> None:
        super(myIRBlock1, self).__init__()
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
            myIRBlock2(16, 6, 24, 2),
            myIRBlock1(24, 6),
            myIRBlock2(24, 6, 32, 2),
            myIRBlock1(32, 6),
            myIRBlock1(32, 6),
            myIRBlock2(32, 6, 64, 2),
            myIRBlock1(64, 6),
            myIRBlock1(64, 6),
            myIRBlock1(64, 6),
            myIRBlock2(64, 6, 96, 1),
            myIRBlock1(96, 6),
            myIRBlock1(96, 6),
            myIRBlock2(96, 6, 160, 2),
            myIRBlock1(160, 6),
            myIRBlock1(160, 6),
            myIRBlock2(160, 6, 320, 1)
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


### SE: Squeeze and Excitation
### NL: Non Linear
### HS: Hard Swish activation function
### RE: Rectified linear unit
### NBN: no batchnorm

class mySE(nn.Module):
    def __init__(self, ioc: int) -> None:
        super(mySE, self).__init__()
        mc = int(ioc/2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fcrelu = nn.Sequential(
            nn.Linear(ioc, mc),
            nn.ReLU(inplace = True)
        )

        self.fcrelu6 = nn.Sequential(
            nn.Linear(mc, ioc),
            nn.ReLU6(inplace = True)
        )
    def forward(self, x: torch.Tensor):
        x0 = x.clone()
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fcrelu(x)
        x = self.fcrelu6(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        return x0 * x.expand_as(x0)


class myBtN(nn.Module): ## Bottleneck in mobilenet upsize the channel_num the desize, unlike the btn in resnet
    def __init__(self, ic: int, mc: int, oc: int, ks: int, strid: int, has_se: bool, act_func: str):
        super(myBtN, self).__init__()
        self.strid = strid
        if act_func == 're':
            self.act_func = nn.ReLU
        elif act_func == 'hs':
            self.act_func = nn.Hardswish
        else:
            raise ValueError(f'Value of act_func must be either hs or re, got {act_func}')
        
        pad = int((ks - 1)/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, mc, 1, 1, 0),
            nn.BatchNorm2d(mc),
            self.act_func(inplace = True)
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(mc, mc, ks, strid, pad, groups = mc),
            nn.BatchNorm2d(mc),
            self.act_func(inplace = True)
        )
        if has_se:
            self.se = mySE(mc)
        else:
            self.se = nn.Identity()

        self.conv2 = nn.Sequential(
            nn.Conv2d(mc, oc, 1, 1, 0),
            nn.BatchNorm2d(oc)
        )


    def forward(self, x: torch.Tensor):
        x0 = x.clone()
        x = self.conv1(x)
        x = self.dw1(x)
        x = self.se(x)
        x = self.conv2(x)
        if x0.shape == x.shape:
            x = x + x0
        else:
            pass

        return x

        

### mbnv3 small architecture: https://camo.githubusercontent.com/0479f4e2f5b23803538f3aa5bf97b03e478f3b9f0362a8aca3b78e0ca8273c2f/68747470733a2f2f692e696d6775722e636f6d2f4264624d3758702e706e67
class myMobileNetV3_small(nn.Module):
    def __init__(self, num_classes: int  = 6):
        super(myMobileNetV3_small, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace = True)
        )
        self.btns = nn.Sequential(
            myBtN(16, 16, 16, 3, 2, True, 're'),
            myBtN(16, 72, 24, 3, 2, False, 're'),
            myBtN(24, 88, 24, 3, 1, False, 're'),
            myBtN(24, 96, 40, 5, 2, True, 'hs'),
            myBtN(40, 240, 40, 5, 1, True, 'hs'),
            myBtN(40, 240, 40, 5, 1, True, 'hs'),
            myBtN(40, 120, 48, 5, 1, True, 'hs'),
            myBtN(48, 144, 48, 5, 1, True, 'hs'),
            myBtN(48, 288, 96, 5, 2, True, 'hs'),
            myBtN(96, 576, 96, 5, 1, True, 'hs'),
            myBtN(96, 576, 96, 5, 1, True, 'hs'),
        )


### mbnv3 large architecture: https://camo.githubusercontent.com/219162cc28a4de064f2434ad70604795e96c54206204dd0337d200d704690edf/68747470733a2f2f692e696d6775722e636f6d2f397757453647502e706e67
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0),
            mySE(576),
            nn.Hardswish(inplace = True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnbn1 = nn.Sequential(
            nn.Conv2d(576, 1280, 1, 1, 0),
            nn.Hardswish(inplace = True)
        )
        self.convnbn2 = nn.Conv2d(1280, num_classes, 1, 1, 0)
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.btns(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.convnbn1(x)
        x = self.convnbn2(x)
        x = x.squeeze()
        return x

class myMobileNetV3_large(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(myMobileNetV3_large, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace = True)
        )

        self.btns = nn.Sequential(
            myBtN(16, 16, 16, 3, 1, 0, 're'),
            myBtN(16, 64, 24, 3, 2, 0, 're'),
            myBtN(24, 72, 24, 3, 1, 0, 're'),
            myBtN(24, 72, 40, 5, 2, 1, 're'),
            myBtN(40, 120, 40, 5, 1, 1, 're'),
            myBtN(40, 120, 40, 5, 1, 1, 're'),
            myBtN(40, 240, 80, 3, 2, 0, 'hs'),
            myBtN(80, 200, 80, 3, 1, 0, 'hs'),
            myBtN(80, 184, 80, 3, 1, 0, 'hs'),
            myBtN(80, 184, 80, 3, 1, 0, 'hs'),
            myBtN(80, 480, 112, 3, 1, 1, 'hs'),
            myBtN(112, 672, 112, 3, 1, 1, 'hs'),
            myBtN(112, 672, 160, 5, 2, 1, 'hs'),
            myBtN(160, 960, 160, 5, 1, 1, 'hs'),
            myBtN(160, 960, 160, 5, 1, 1, 'hs')
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, 1, 1, 0),
            nn.Hardswish(inplace = True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.convnbn1 = nn.Sequential(
            nn.Conv2d(960, 1280, 1, 1, 0),
            nn.Hardswish(inplace = True)
        )
        self.linnbn2 = nn.Sequential(
            nn.Conv2d(1280, num_classes, 1, 1, 0)
        )
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.btns(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.convnbn1(x)
        x = self.convnbn2(x)
        x = x.squeeze()
        return x

model = myMobileNetV3_small()
print(model)
summary(model, (3, 224, 224), device = 'cpu')