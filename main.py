import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder
import argparse
import torchvision.transforms as T
from torchsummary import summary

from models.myresnet import *
from models.mymobilenet import *
from utils.dataload import *
from utils.actions import *

datasource = DataSource(root = './dataset/')

# model = myresnet18()
# model = myresnet34()
# model = myresnet50()
# model = myresnet101()
# model = myresnet152()
# model = myMobileNetV1()
model = myMobileNetV2()
model = myMobileNetV3_small()
model = myMobileNetV3_large()

summary(model, (3, 224, 224), device = 'cpu')

trainset = datasource.get_trainset(root = 'seg_train/seg_train/')
testset, valset = datasource.get_test_valset(root = 'seg_test/seg_test/')

Train(model = model, trainset = trainset, valset = valset)
Test(model = model, testset = testset)
