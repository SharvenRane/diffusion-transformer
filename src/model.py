"""
Diffusion Transformer - Model Architecture
"""
import torch
import torch.nn as nn
import timm
from einops import rearrange


class DiffusionTransformer(nn.Module):
    """
    Main model for diffusion transformer.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            features_only=True
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.feature_info[-1]["num_chs"], config.num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features[-1])
        return out

    def extract_features(self, x):
        features = self.encoder(x)
        pooled = nn.functional.adaptive_avg_pool2d(features[-1], 1)
        return pooled.flatten(1)


def build_model(config):
    model = DiffusionTransformer(config)
    if config.get("checkpoint"):
        state = torch.load(config.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
    return model

# update 10
