import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name='resnet18', num_classes=4, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Model not supported. Choose 'resnet18' or 'mobilenet_v2'")
    
    return model

if __name__ == "__main__":
    model = get_model('resnet18', 4)
    print(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
