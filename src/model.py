import imp
import torch
import torch.nn as nn
import torchvision
import os
from torch.backends import cudnn
import matplotlib.pyplot as plt

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
        
        
        # we will save the conv layer weights in this list
        self.model_weights =[]
        #we will save the 49 conv layers in this list
        self.conv_layers = []# get all the model children as list
        self.model_children = list(self.model.children())#counter to keep count of the conv layers
        counter = 0#append all the conv layers and their respective wights to the list
        for i in range(len(self.model_children)):
            if type(self.model_children[i]) == nn.Conv2d:
                counter+=1
                self.model_weights.append(self.model_children[i].weight)
                self.conv_layers.append(self.model_children[i])
            elif type(self.model_children[i]) == nn.Sequential:
                for j in range(len(self.model_children[i])):
                    for child in self.model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter+=1
                            self.model_weights.append(child.weight)
                            self.conv_layers.append(child)
        print(f"Total convolution layers: {counter}")
        print("conv_layers")
        
        # print(f"\nMODEL SUMMARY:\n\n{self.model}\n")

    def load_checkpoint(self, inputs_ckpt=None):
        if inputs_ckpt:
            self.inputs_ckpt = inputs_ckpt
        states = torch.load(self.inputs_ckpt, map_location=self.device)
        self.model.load_state_dict(states)
    
    def save_checkpoints(self, inputs_ckpt=None):
        if inputs_ckpt:
            self.inputs_ckpt = inputs_ckpt
        torch.save(self.model.state_dict(), self.inputs_ckpt)
        
    def visualize_feature_maps(self, image):
        outputs = []
        names = []
        processed = []  # Initialize the list to store feature maps

        for layer in self.conv_layers[0:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))

            for feature_map in outputs:
                feature_map = feature_map.squeeze(0)
                gray_scale = torch.sum(feature_map, 0)
                gray_scale = gray_scale / feature_map.shape[0]
                processed.append(gray_scale.data.cpu().numpy())

        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i % 20 + 1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(str(names[0]), fontsize=30)

        plt.savefig('feature_maps.jpg', bbox_inches='tight')
