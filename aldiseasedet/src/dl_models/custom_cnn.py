import torch.nn as nn

def get_model(num_classes: int) -> nn.Module:
    """Tiny custom CNN for Apple leaf classification.
    Architecture:
        Conv2d(3, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        Conv2d(64, 128, 3, padding=1) -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d(1)
        Flatten -> Linear(128, num_classes)
    Returns:
        A torch.nn.Module ready for training/evaluation.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),

        nn.Flatten(),
        nn.Linear(128, num_classes),
    )
    return model
