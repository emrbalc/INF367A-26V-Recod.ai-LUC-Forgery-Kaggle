import torch
import torch.nn as nn

class SegNeXtSegmenter(nn.Module):
    def __init__(self, out_ch: int = 1):
        super().__init__()
        self.out_ch = out_ch
        # Minimal learnable layer so optimizer has params
        self.head = nn.Conv2d(3, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        return self.head(x)