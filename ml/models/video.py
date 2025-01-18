import torch
import torch.nn as nn
from torchvision import models

class VideoEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.resnet = models.video.r3d_18(pretrained=True)
        
        for param in self.resnet.parameters():
            param.requires_grad = False
        
            
        num_features = self.resnet.fc.in_features
        
        # replace the fully connected layer
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.resnet(x)
        