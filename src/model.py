import imp
import torch
import torch.nn as nn
import torchvision
import os
from torch.backends import cudnn
import matplotlib.pyplot as plt

cudnn.benchmark = True

class ResNet(nn.Module):
    def __init__(self, device=None, path = "checkpoints/resnet.pth", nbr_classes=2, number_conv_layers=2):
        super(ResNet, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        n_classes = nbr_classes
        self.model.fc = nn.Linear(num_ftrs, n_classes)
        
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.inputs_ckpt = os.path.join("checkpoints", "resnet.pth")
        
        # Initialize a new model with the same architecture as self.model
        new_model = []
        conv_layer_count = 0
        last_conv_layer = None  # Initialize with None
        for name, child in self.model.named_children():
            if conv_layer_count < number_conv_layers:
                if any(isinstance(module, nn.Conv2d) for module in child.modules()):
                    # If the child contains a convolutional layer, add it to the new model
                    new_model.append(child)
                    conv_layer_count += 1
                    last_conv_layer = child
                else:
                    # If the child does not contain a convolutional layer, you can skip it
                    pass
            else:
                # If you've reached the desired number of convolutional layers, stop adding
                break
        # add (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
        # new_model.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # add flatten layer :
        # new_model.append(nn.Flatten())
        
        new_model = nn.Sequential(*new_model).to(self.device)
        
        # Calculate the number of input features for the Linear layer
        print("last_conv_layer : ", last_conv_layer)
        shape = None
        for name, children in last_conv_layer.named_children():
            # check if this children is the last one if so continue
            
            for name_2, children_2 in children.named_children():
                if name_2 == "relu":
                    continue
                elif name_2 == "downsample":
                    for name_3, children_3 in children_2.named_children():
                        # print("\n\nMODULE name : ", name_3)
                        # print("DIR : ", dir(children_3))
                        try:
                            print("num features : ", children_3.num_features)
                            num_ftrs = children_3.num_features
                        except:
                            pass
                        # print("weight : ", children_3.weight.shape)
                else: 
                    # print("\n\nMODULE name : ", name_2)
                    # print("DIR : ", dir(children_2))
                    try:
                        # print("num features : ", children_2.num_features)
                        num_ftrs = children_2.num_features
                    except:
                        pass
                    # print("weight : ", children_2.weight.shape)
        random_input = torch.randn(32, 3, 128, 128).to(self.device)
        output = new_model(random_input)
        # print("output shape : ", output.shape)
        # num_ftrs = output.shape[1] # output.shape[1] * output.shape[2] * output.shape[3]
        n_classes = nbr_classes
        print("attributes of last_conv_layer : ", dir(last_conv_layer))
        
        num_ftrs = 8
        new_model.add_module("fc", nn.Linear(num_ftrs, n_classes))
        
        # Now, new_model includes the last Linear layer with the appropriate input features
        print("Original Model : ", self.model)
        print("\n\n\n\nNew model : ", new_model)
        self.model = new_model
        self.model.to(self.device)
        
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
        self.last_conv_layer = last_conv_layer

    
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


if __name__ == "__main__":
    model = ResNet()