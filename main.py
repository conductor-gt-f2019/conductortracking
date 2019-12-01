from Model import CT_Model
from train import train_model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

model = CT_Model(2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)