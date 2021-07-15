import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torchvision import models

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.model.fc = nn.Linear(256, config.num_labels)


    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=-1)
        
        return x


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, config.num_labels)


    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=-1)
        
        return x
        
        
class ResNet18_not_pre(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, config.num_labels)


    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=-1)
        
        return x


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.fc = nn.Linear(512, config.num_labels)


    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=-1)
        
        return x
        
        
