from torchvision import transforms

def get_resnet_transforms():
    return transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])