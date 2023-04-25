import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import Inception3
from torchsummary import summary

### CURRENTLY UNAVAILABLE, CHECK IN LATER FOR FUTURE FIXES

class BIInceptionV3(Inception3):
    def __init__(self, num_classes: int = 6, aux_logits: bool = True, transform_input: bool = False, inception_blocks= None, init_weights= None, dropout: float = 0.5) -> None:
        super(BIInceptionV3, self).__init__(num_classes, aux_logits, transform_input, inception_blocks, init_weights, dropout)


class myInceptionV3(nn.Module):
    def __init__(self) -> None:
        super(myInceptionV3, self).__init__()


model = BIInceptionV3(init_weights = True)
summary(model, (3, 299, 299), device = 'cpu')