
import torchvision.models as models
import torch.nn as nn

def get_model(model_name, num_classes=3, pretrained=True):
    """
    Loads a pre-trained model and adapts it for the given number of classes.

    Args:
        model_name (str): The name of the model to load (e.g., 'resnet18').
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to load pre-trained weights.

    Returns:
        torch.nn.Module: The adapted model.
    """
    model = None
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # Modify the first layer to accept 1 channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer for our number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        # Modify the first layer
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Modify the final layer
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        # Modify the first layer
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the final layer
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model
