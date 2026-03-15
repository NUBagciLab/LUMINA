import torch
import torchvision

def get_model(name = 'resnet50', num_classes=1):
    if 'resnet' in name:
        model = torchvision.models.get_model(name, weights="IMAGENET1K_V2", num_classes=1000)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes, bias=True)
    elif 'efficientnet' in name:
        model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes, bias=True)
    elif 'densenet' in name:
        model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes, bias=True)
    elif 'swin' in name:
        model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
        model.head = torch.nn.Linear(model.head.in_features, num_classes, bias=True)
    else:
        raise Exception("Model is not supported")
    return model