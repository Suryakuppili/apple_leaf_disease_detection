import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def get_model(num_classes: int) -> nn.Module:
    """MobileNetV2 wrapper with strictly ImageNet pretrained weights.

    Raises:
        RuntimeError: If the pretrained weights cannot be downloaded or loaded.
    """
    try:
        weights = MobileNet_V2_Weights.IMAGENET1K_V1
        model = mobilenet_v2(weights=weights)
    except Exception as e:
        raise RuntimeError(f"Failed to load MobileNetV2 pretrained weights: {e}") from e

    # Replace the final classifier layer to match the number of classes.
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
