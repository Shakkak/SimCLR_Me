import torchvision


import torchvision
import torch.nn as nn

def get_resnet(name, pretrained=False):
    # Define the available ResNet models
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    
    # Get the selected ResNet model
    model = resnets[name]

    # Modify the first convolution layer to accept 128x128 images
    model.conv1 = nn.Conv2d(
        in_channels=3,      # Input channels (3 for RGB)
        out_channels=64,    # Output channels
        kernel_size=(7, 7), # Kernel size
        stride=(2, 2),      # Stride
        padding=(3, 3),     # Padding
        bias=False          # No bias for BatchNorm
    )

    # Modify the average pooling layer to accommodate the smaller input size
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to output size of (1, 1)

    return model
