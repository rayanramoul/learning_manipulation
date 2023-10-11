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
    def __init__(self, model, feature_layer=None, target_layer=2, use_cuda=False):
        self.model = model
        # GradCam have to be run with fixed weights
        self.model.eval() 
        self.convlayer = feature_layer
        self.workflow = ModelOutputs(self.model, self.convlayer, target_layer)  
        self.use_cuda = use_cuda

    def forward(self, image):
        return self.model(image)

    def __call__(self, input_img, target_category=None):
        # If we're using a gpu take the image on gpu
        if self.use_cuda:
            input_img = input_img.cuda()

        # Doing a forward pass with the image
        features, output = self.workflow(input_img)
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
            
        # We consider only the gradient of the targeted class so we set the output of others to zero
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.use_cuda:
            one_hot = one_hot.cuda()
        
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

