import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,return_indices=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2 ,return_indices=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2 ,return_indices=True),
        )
        self.rev_features = nn.Sequential(
            nn.ConvTranspose2d( 64, 3, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ConvTranspose2d( 192, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxUnpool2d(kernel_size=3, stride=2),
            nn.ConvTranspose2d( 384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxUnpool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.features:
          if layer._get_name()=='MaxPool2d':
            x,_ = self.features(x)
          else:
            x =self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


an=AlexNet()

# Download VGG16 model file to path ./models/alexnet/model.pth
state_dict = torch.load('./models/alexnet/model.pth')

#Loading Parameters and initialize them for both forward and backward layers
an.load_state_dict(state_dict,strict=False)
for i,j in zip(an.features,an.rev_features):
  if i._get_name()=='Conv2d':
    j.load_state_dict({'weight':i.weight,'bias':torch.zeros_like(j.bias)})
