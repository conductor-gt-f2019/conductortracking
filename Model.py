from torch import nn as nn
from torchvision import models
import torch.nn.functional as F

class CT_Model(nn.Module):
    def __init__(self, n_classes):
        super(CT_Model, self).__init__()
        self.model = models.resnet18(True)
        self.in_features = self.model.fc.in_features
        self.n_classes = n_classes
        self.model.fc = nn.Linear(self.in_features, self.n_classes)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=self.n_classes)