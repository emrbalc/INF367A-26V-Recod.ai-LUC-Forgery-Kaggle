import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class SegNeXtSegmenter(nn.Module):

    def __init__(self, out_ch: int = 1):
        super().__init__()

        # SegNeXt backbone (MSCAN)
        self.backbone = timm.create_model(
            "segnext_small",
            pretrained=True,
            features_only=True
        )

        channels = self.backbone.feature_info.channels()

        # decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(channels[-1], 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, out_ch, 1)
        )


    def forward(self, x):

        features = self.backbone(x)

        x = features[-1]

        x = self.decoder(x)

        x = F.interpolate(
            x,
            scale_factor=32,
            mode="bilinear",
            align_corners=False
        )

        return x
