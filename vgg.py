import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, rev_features: nn.Module, num_classes: int = 1000, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.rev_features = rev_features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
            if layer._get_name() == 'MaxPool2d':
                x, _ = layers(x)
            else:
                x = layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers, rev_layers = [], []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2,
                                    stride=2, return_indices=True)]
            rev_layers += [nn.MaxUnpool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            rev_layers += [nn.ConvTranspose2d(v, in_channels,
                                              kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), nn.Sequential(*rev_layers)


vgg16_config = [64, 64, "M", 128, 128, "M", 256, 256,
                256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

vgg16 = VGG(*make_layers(vgg16_config))

# Download VGG16 model file to path ./models/vgg16/model.pth
state_dict = torch.load('./models/vgg16/model.pth')

# Loading Parameters and initialize them for both forward and backward layers
vgg16.load_state_dict(state_dict, strict=False)
for i, j in zip(vgg16.features, vgg16.rev_features):
    if i._get_name() == 'Conv2d':
        j.load_state_dict(
            {'weight': i.weight, 'bias': torch.zeros_like(j.bias)})
