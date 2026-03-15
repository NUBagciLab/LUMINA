import torch
import torchvision
from torch import nn

# def get_model(name = 'resnet50', num_classes=1):
#     if 'resnet' in name:
#         model = torchvision.models.get_model(name, weights="IMAGENET1K_V2", num_classes=1000)
#         model.fc = torch.nn.Linear(model.fc.in_features, num_classes, bias=True)
#     elif 'efficientnet' in name:
#         model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
#         model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes, bias=True)
#     elif 'densenet' in name:
#         model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
#         model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes, bias=True)
#     elif 'swin' in name:
#         model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
#         model.head = torch.nn.Linear(model.head.in_features, num_classes, bias=True)
#     else:
#         raise Exception("Model is not supported")
#     return model

class get_model(nn.Module):
    def __init__(self, name="resnet50", num_classes=1, view = 2, fusion_hidden=128):
        super(get_model, self).__init__()
        if 'resnet' in name:
            model = torchvision.models.get_model(name, weights="IMAGENET1K_V2", num_classes=1000)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
        elif 'efficientnet' in name:
            model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
        elif 'densenet' in name:
            model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()
        elif 'swin' in name:
            model = torchvision.models.get_model(name, weights="IMAGENET1K_V1", num_classes=1000)
            in_features = model.head.in_features
            model.head = nn.Identity()
        else:
            raise Exception("Model is not supported")
        self.view = view
        self.backbones = model

        # Fusion MLP after concatenating 4 view features
        self.classifier = nn.Sequential(
            nn.Linear(view * in_features, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, num_classes)
        )

    def forward(self, x:list):
        f = torch.cat([self.backbones(x[i]) for i in range(self.view)], dim=1)
        return self.classifier(f)