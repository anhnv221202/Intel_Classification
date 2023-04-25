import torch
import torchvision.transforms as T
import os
import numpy as np
import cv2
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset, random_split

train_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.RandomHorizontalFlip(p = 0.3),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_val_tran = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def valid(path: str):
    try:
        img = cv2.imread(path)
        condition1 = (img.shape == (150, 150, 3))
        condition2 = img.any()
        return condition1 and condition2
    except Exception:
        return False
    

class DataSource():
    def __init__(self, root: str):
        self.root = root


    def get_trainset(self, root: str, transformer: T.transforms = train_transform, filter_func = valid):
        root = os.path.join(self.root, root)
        trainset = ImageFolder(root = root, transform = transformer, is_valid_file = filter_func)
        return trainset
    
    
    def get_test_valset(self, root: str, transformer: T.transforms = test_val_tran, filter_func = valid):
        root = os.path.join(self.root, root)
        test_and_val_set = ImageFolder(root = root, transform = transformer, is_valid_file = filter_func)
        testlen = len(test_and_val_set)//2
        vallen = len(test_and_val_set) - testlen
        testset, valset = random_split(dataset = test_and_val_set, lengths = [testlen, vallen])
        return testset, valset