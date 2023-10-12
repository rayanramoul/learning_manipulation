import cv2
import torch
import numpy as np
from torchvision import transforms
from skimage.filters import threshold_otsu
from PIL import Image

import scipy.ndimage as ndimage
import scipy.spatial as spatial

class ExtractActiv:
    """
    Allow extraction of each output of layer 
    And Attach it to a function for saving gradioents
    """
    def __init__(self, model, target):
        self.gradients = [] 
        self.model = model
        self.target = target

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        out = []
        self.gradients = []
        for name, layer in self.model._modules.items():
            # Doing a forward pass
            x = layer(x)
            if name in self.target:
                #  Register hook on last conv layer so we can get its gradient later
                x.register_hook(self.save_gradient) 
                out += [x]
        return out, x

class ModelOutputs:
    """
    Process forward pass and return :
    - Network output
    - Activation from wanted last conv layer
    - Gradients from wanted target layer
    """
    def __init__(self, model, feature_module, target):
        self.model = model
        self.feature = feature_module
        self.feature_extractor = ExtractActiv(self.feature, target)

    def get_gradients(self):
        return self.feature_extractor.gradients
    
    def __call__(self, inpu):
        # if there is gpu we take the image on gpu
        if torch.cuda.is_available():
            inpu = inpu.cuda()
        
        activations = []
        for name, layer in self.model._modules.items():
            if layer == self.feature:
                activations, inpu = self.feature_extractor(inpu)
            elif "avgpool" in name.lower():
                inpu = layer(inpu)
                inpu = inpu.view(inpu.size(0), -1)
            else:
                inpu = layer(inpu)
        
        return activations, inpu



class GradCam:
    """
        Class for computing forward pass on a given model and extract Heatmap.
    """
    def __init__(self, model, feature_layer=None, target_layer=2, device=None):
        self.model = model
        # GradCam have to be run with fixed weights
        self.model.eval() 
        self.convlayer = feature_layer
        self.workflow = ModelOutputs(self.model, self.convlayer, target_layer)
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

    def forward(self, image):
        return self.model(image)

    def __call__(self, input_img, target_category=None):
        # If we're using a gpu take the image on gpu
        
        input_img.to(self.device)

        # Doing a forward pass with the image
        features, output = self.workflow(input_img)
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
            
        # We consider only the gradient of the targeted class so we set the output of others to zero
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        one_hot = one_hot.to(self.device)
        output = output.to(self.device)
        one_hot = torch.sum(one_hot * output)

        # We calculate gradient then we extract it with the appropriate class
        self.convlayer.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.workflow.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        # We do a pooling over width and height of last feature layer
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # We upscale the resulting heatmap
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:][::-1])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def preprocess_image(img, preprocess):
    """
    Normalizing an image using mean and std of VOC dataset.
    """
    numpy_image = img.copy()
    torch_img = Image.fromarray(np.uint8(numpy_image)).convert('RGB') 
    print("type torch_img : ", type(torch_img))
    preprocessed_img  = preprocess(torch_img)
    return preprocessed_img.unsqueeze(0)

def eval_image(gradcam, path, target_category=4, transform_pipeline=None):
    """
    Evaluate an Image with GradCAM algorithm
    Input :
        path : path for the image to predict heatmap from
        target_category : which category prediction we're interested in
    Output :
        input_img : the image after preprocessing
        grayscale_cam : heatmap of relevant pixels in picture
        cam : Image + heatmap
        img : original image
    """

    img = cv2.imread(path, 1)
    #img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    print("img shape : ", img.shape)
    img = img[:, :, ::-1]
    input_img = preprocess_image(img, transform_pipeline)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    grayscale_cam = gradcam(input_img, target_category)
      
    # resize grayscale cam to original image size
    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    
    cam = show_cam_on_image(img, grayscale_cam)
    
    return input_img, grayscale_cam, cam, img

def show_cam_on_image(img, mask):
    """
    Input :
        mask : Heatmap from GradCAM
        img : Original Image
    Output :
        cam : The image with a heatmap mask on it
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_bdbox_from_heatmap(heatmap, threshold=0.2, smooth_radius=20):
    """
    Function to extract bounding boxes of objects in heatmap
    Input :
        Heatmap : matrix extracted with GradCAM. 
        threshold : value defining the values we consider , increasing it increases the size of bounding boxes.
        smooth_radius : radius on which each pixel is blurred. 
    Output :
        returned_objects : List of bounding boxes, N_objects * [ xmin, xmax, ymin, ymax, width, height ]
    """

    # If heatmap is all zeros i initialize a default bounding box which wraps entire image
    xmin = 0
    xmax = heatmap.shape[1]
    ymin = 0
    ymax = heatmap.shape[0]
    width = xmax-xmin
    height = ymax-ymin
    
    returned_objects = []

    # Count if there is any "hot" value on the heatmap
    count = (heatmap > threshold).sum() 
    
    # Blur the image to have continuous regions
    heatmap = ndimage.uniform_filter(heatmap, smooth_radius)

    # Threshold the heatmap with 1 for values > threshold and 0 else
    thresholded = np.where(heatmap > threshold, 1, 0)

    # Apply morphological filter to fill potential holes in the heatmap
    thresholded =  ndimage.morphology.binary_fill_holes(thresholded)

    # Detect all independant objects in the image
    labeled_image, num_features = ndimage.label(thresholded)
    objects = ndimage.measurements.find_objects(labeled_image)
    
    # We loop in each object ( if any is detected ) and append it to a global list
    if count > 0:
        for obj in objects:
            x = obj[1]
            y = obj[0]
            xmin = x.start
            xmax = x.stop
            ymin = y.start
            ymax = y.stop

            width = xmax-xmin
            height = ymax-ymin
            
            returned_objects.append([xmin, xmax, ymin, ymax, width, height])
    else:
        returned_objects.append([xmin, xmax, ymin, ymax, width, height])
    return returned_objects
