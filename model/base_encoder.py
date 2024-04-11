import torch.nn as nn

from torchvision import models
from .ViT.mae import VisionTransformers as FeatureExtractor
class MAEEncoder(nn.Module):

    def __init__(self, MAEConfigs):
        super(MAEEncoder, self).__init__()
        self.model = FeatureExtractor(**MAEConfigs)
    def forward(self, x):
        batch_size, seqlen, H, W = x.shape
        x = x.reshape(batch_size * seqlen, 1, H, W)
        x = self.model(x)
        return x.reshape(batch_size, seqlen, -1)

class RESNETEncoder(nn.Module):

    def __init__(self, model_type, feature_len):
        super(RESNETEncoder, self).__init__()
        if model_type == 'resnet18':
            self.model = models.resnet18(pretrained=False)
        elif model_type == 'resnet34':
            self.model = models.resnet34(pretrained=False)
        elif model_type == 'resnet50':
            self.model = models.resnet50(pretrained=False)
        elif model_type == 'resnet101':
            self.model = models.resnet101(pretrained=False)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=3, bias=False)
        num_final_in = self.model.fc.in_features
        self.model.fc = nn.Linear(num_final_in, feature_len)

    def forward(self, x):
        batch_size, seqlen, H, W = x.shape
        x = x.reshape(batch_size * seqlen, 1, H, W)
        x = self.model(x)
        return x.reshape(batch_size, seqlen, -1)