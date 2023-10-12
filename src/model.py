import torch
import torch.nn as nn
import torchvision
import os
from torch.backends import cudnn

cudnn.benchmark = True

class ResNet():
    def __init__(self, device=None, path = "checkpoints/resnet.pth", nbr_classes=2):
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        n_classes = nbr_classes
        self.model.fc = nn.Linear(num_ftrs, n_classes)
        
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)
        self.model = self.model.to(self.device)
        self.inputs_ckpt = os.path.join("checkpoints", "resnet.pth")
        print(f"\nMODEL SUMMARY:\n\n{self.model}\n")

    def load_checkpoint(self, inputs_ckpt=None):
        if inputs_ckpt:
            self.inputs_ckpt = inputs_ckpt
        states = torch.load(self.inputs_ckpt, map_location=self.device)
        self.model.load_state_dict(states)
    
    def save_checkpoints(self, inputs_ckpt=None):
        if inputs_ckpt:
            self.inputs_ckpt = inputs_ckpt
        torch.save(self.model.state_dict(), self.inputs_ckpt)
        
